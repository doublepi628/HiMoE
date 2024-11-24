import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphLearning(nn.Module):
    def __init__(self, args):
        super(GraphLearning, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(2, 8, bias=True)
        self.fc2 = nn.Linear(8, 1, bias=True)
        self.mask = nn.Parameter(torch.full((args.adj.shape[0], args.adj.shape[0]), 0.0), requires_grad=True)
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
    def forward(self, dist, relat):
        fusion_adj = torch.stack((dist, relat), dim=-1).to(self.args.device)
        fusion_adj = self.fc2(self.fc1(fusion_adj))
        gated_adj = torch.mul(fusion_adj[:,:,0], torch.tanh(self.mask))
        return gated_adj + gated_adj.T

class HiGCN(nn.Module):
    def __init__(self, args, in_features, out_features):
        super(HiGCN, self).__init__()
        self.adj = args.adj
        self.in_features = in_features
        self.out_features = out_features
        self.fc_weight = nn.Linear(in_features, out_features, bias=True)
        self.fc_self = nn.Linear(in_features, out_features, bias=False)
        self.fc_emb = nn.Linear(in_features, 32, bias=False)
        self.graph = GraphLearning(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_weight.reset_parameters()
        self.fc_self.reset_parameters()
        self.fc_emb.reset_parameters()

    def forward(self, x, adj_mask):
        dynamic_adj = self.dynamic_graph(x)
        learnable_adj = self.graph(self.adj, self.normalize_graph(dynamic_adj))
        if adj_mask == None:
            filtered_graph = learnable_adj
        else:
            filtered_graph = torch.mul(learnable_adj, adj_mask)
        graph_convolution = self.fc_weight(torch.matmul(filtered_graph, x)) + self.fc_self(x)
        return graph_convolution

    def normalize_graph(self, adj):
        binary_adj = (adj > 0).float()
        degree = torch.sum(binary_adj, dim=1) + 1e-8
        D_inv_sqrt = torch.diag(degree.pow(-0.5))
        return torch.matmul(D_inv_sqrt, torch.matmul(adj, D_inv_sqrt))

    def dynamic_graph(self, x):
        x = self.fc_emb(x)
        mean = torch.mean(x)  
        std = torch.std(x) 
        standard_x = ((x - mean) / (std + 1e-8)).detach()
        standard_xt = standard_x.transpose(1, 2)
        dynamic_adj = torch.bmm(standard_x, standard_xt)
        dynamic_adj = torch.sum(dynamic_adj, dim=0)
        dynamic_adj.fill_diagonal_(0.0)
        return torch.softmax(dynamic_adj, dim=1)
    

class Expert(nn.Module):
    def __init__(self, args):
        super(Expert, self).__init__()
        self.gcn1 = HiGCN(args, args.gcn["in_channel"], args.gcn["hidden_channel"])
        self.gcn2 = HiGCN(args, args.gcn["hidden_channel"], args.gcn["out_channel"])
        self.tcn1 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.tcn2 = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.fc = nn.Linear(args.gcn["out_channel"], args.y_len)
        self.activation = nn.GELU()
        self.args = args

    def forward(self, data, adj_mask):
        N = self.args.adj.shape[0]
        x = data.reshape((-1, N, self.args.gcn["in_channel"]))     # [bs, N, feature]
        x = F.relu(self.gcn1(x, adj_mask))                         # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["hidden_channel"]))    # [bs * N, 1, feature]
        x = self.tcn1(x)                                           # [bs * N, 1, feature]
        x = x.reshape((-1, N, self.args.gcn["hidden_channel"]))    # [bs, N, feature]
        x = self.gcn2(x, adj_mask)                                 # [bs, N, feature]
        x = x.reshape((-1, 1, self.args.gcn["out_channel"]))
        x = self.tcn2(x)    
        x = x.reshape((-1, N, self.args.gcn["out_channel"]))
        x = self.fc(self.activation(x))
        return x


class HiMoE(nn.Module):
    def __init__(self, args):
        super(HiMoE, self).__init__()
        self.args = args
        self.experts = nn.ModuleList([Expert(args) for _ in range(args.k_experts)])
        self.bound = nn.Parameter(torch.tensor(-5.0), requires_grad=True)
        self.deltas = nn.ParameterList([nn.Parameter(torch.tensor(10 / args.k_experts), requires_grad=True) for _ in range(args.k_experts - 2)])
        self.masks = nn.Parameter(torch.randn(args.k_experts, 1, args.adj.shape[0], 1), requires_grad=True)
        self.mask = nn.Parameter(torch.ones(args.k_experts, 1, args.adj.shape[0], 1), requires_grad=True)
        self.gelu = nn.GELU()   

    def forward(self, data):
        # data: [batch size, N, T]
        means = torch.mean(data, dim=(0,2)) # [N]
        scaled_means = (means - torch.mean(means)) / (torch.std(means)+1e-7) # [N]
        results = [self.experts[self.args.k_experts - 1](data, None)] # [batch size, N, T']
        bound = self.bound
        for i in range(self.args.k_experts - 1):
            if i >= 1:
                bound = bound + self.gelu(self.deltas[i-1])
            selected = 1 / (1 + torch.exp(-(scaled_means - bound) * self.args.k_dilate)) # [N]
            adj_mask = torch.matmul(selected.unsqueeze(1), selected.unsqueeze(0))
            predict = self.experts[i](data, adj_mask)
            results.append(predict)
        result = torch.sum(torch.stack(results, dim=0) * torch.sigmoid(self.masks), dim=0)
        return result