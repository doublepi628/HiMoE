import argparse, random, re, json, logging, sys, os
import numpy as np, os.path as osp
import torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader
from train import train
from validate import validate
from graph import scaled_Laplacian, cheb_polynomial, stsgcn_adj


def seed_set(seed=0):
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True


def json2args(args):
    with open(args.conf, "r") as f:
        s = f.read()
        s = re.sub('\s',"", s)
    info = json.loads(s)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    for key in info:
        vars(args)[key] = info[key]
    args.path = osp.join("logs", args.dataset, args.model_name, args.time)
    del info


def init_log(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(args.log_dir, args.dataset, f"{args.model_name}-{args.time}.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(args.log_dir, args.dataset, f"{args.model_name}-{args.time}.log"))
    vars(args)["logger"] = logger
    return logger


def generate_dataloader(args):
    # IMPORTANT: HiMoE does not require timestamp information, but we have still provided this interface.
    to_torch = lambda x: torch.from_numpy(x).type(torch.FloatTensor).to(args.device)
    args.mu, args.sigma = np.load(args.data_dir + 'dataset/dist.npy')
    train_x = np.load(args.data_dir + 'dataset/train.npz')['x'][:,:,:,0].transpose(0, 2, 1)
    train_y = np.load(args.data_dir + 'dataset/train.npz')['y'][:,:,:,0].transpose(0, 2, 1) * args.sigma + args.mu
    train_tod_x = np.load(args.data_dir + 'dataset/train.npz')['x'][:,:,:,1].transpose(0, 2, 1)
    train_dow_x = np.load(args.data_dir + 'dataset/train.npz')['x'][:,:,:,2].transpose(0, 2, 1)
    val_x = np.load(args.data_dir + 'dataset/val.npz')['x'][:,:,:,0].transpose(0, 2, 1)
    val_y = np.load(args.data_dir + 'dataset/val.npz')['y'][:,:,:,0].transpose(0, 2, 1) * args.sigma + args.mu
    val_tod_x = np.load(args.data_dir + 'dataset/val.npz')['x'][:,:,:,1].transpose(0, 2, 1)
    val_dow_x = np.load(args.data_dir + 'dataset/val.npz')['x'][:,:,:,2].transpose(0, 2, 1)
    test_x = np.load(args.data_dir + 'dataset/test.npz')['x'][:,:,:,0].transpose(0, 2, 1)
    test_y = np.load(args.data_dir + 'dataset/test.npz')['y'][:,:,:,0].transpose(0, 2, 1) * args.sigma + args.mu
    test_tod_x = np.load(args.data_dir + 'dataset/test.npz')['x'][:,:,:,1].transpose(0, 2, 1)
    test_dow_x = np.load(args.data_dir + 'dataset/test.npz')['x'][:,:,:,2].transpose(0, 2, 1)
    train_val_x = np.concatenate([train_x, val_x], axis=0)
    train_val_tod_x = np.concatenate([train_tod_x, val_tod_x], axis=0)
    train_val_dow_x = np.concatenate([train_dow_x, val_dow_x], axis=0)
    train_val_y = np.concatenate([train_y, val_y], axis=0)
    train_val_idx = np.arange(train_val_x.shape[0])
    t = train_val_x.shape[0]
    np.random.shuffle(train_val_idx)
    # train:val:test = 6:2:2
    train_x, val_x = train_val_x[train_val_idx[:int(t*0.75)]], train_val_x[train_val_idx[int(t*0.75):]]
    train_tod_x, val_tod_x = train_val_tod_x[train_val_idx[:int(t*0.75)]], train_val_tod_x[train_val_idx[int(t*0.75):]]
    train_y, val_y = train_val_y[train_val_idx[:int(t*0.75)]], train_val_y[train_val_idx[int(t*0.75):]]
    train_dow_x, val_dow_x = train_val_dow_x[train_val_idx[:int(t*0.75)]], train_val_dow_x[train_val_idx[int(t*0.75):]]
    args.train_loader = DataLoader(TensorDataset(to_torch(train_x), to_torch(train_tod_x), to_torch(train_dow_x), to_torch(train_y)), batch_size=args.batch_size, shuffle=True)
    args.val_loader = DataLoader(TensorDataset(to_torch(val_x), to_torch(val_tod_x), to_torch(val_dow_x), to_torch(val_y)), batch_size=args.batch_size, shuffle=False)
    args.test_loader = DataLoader(TensorDataset(to_torch(test_x), to_torch(test_tod_x), to_torch(test_dow_x), to_torch(test_y)), batch_size=args.batch_size, shuffle=False)


def generate_adj(args):
    adj = np.load(args.data_dir + 'graph/weighted.npy')
    args.adj = torch.from_numpy(adj).type(torch.FloatTensor).to(args.device)
    args.num_nodes = args.adj.shape[0]

def load_weight(args):
    # args.eval_mean_weight = np.load(args.data_dir + 'weight/eval_mean_weight.npy')
    # args.loss_mean_weight = np.load(args.data_dir + 'weight/loss_mean_weight.npy')
    args.eval_mean_ratio = np.load(args.data_dir + 'weight/eval_mean_ratio.npy')
    args.loss_mean_ratio = np.load(args.data_dir + 'weight/loss_mean_ratio.npy')


def main(args):
    generate_dataloader(args)
    generate_adj(args)
    load_weight(args)
    if args.mode != 'test_only':
        train(args)
    validate(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type = str, default = "./conf/himoe.himoe.json")
    parser.add_argument("--dataset", type = str, default = "PEMS04")
    parser.add_argument("--data_dir", type = str, default = "./data/PEMS04/")
    parser.add_argument("--gpuid", type = int, default = 2)
    parser.add_argument("--mode", type=str, default="train&test")
    parser.add_argument("--best_model_path", type=str, default="./")
    args = parser.parse_args()
    json2args(args)
    if not os.path.exists(args.log_dir + args.dataset):
        os.makedirs(args.log_dir + args.dataset)
    if not os.path.exists(args.model_dir + args.dataset):
        os.makedirs(args.model_dir + args.dataset)
    seed_set(args.seed)
    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    args.device = device
    args.logger = init_log(args)
    args.logger.info("params : %s", vars(args))
    main(args)