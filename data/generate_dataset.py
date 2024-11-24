import argparse
import numpy as np
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

def generate_weight(args, data):
    data = data.transpose(1,0)
    mean_val = np.mean(data, axis=1)
    mean_ratio = mean_val / np.mean(mean_val)
    return mean_ratio


def generate_graph_seq2seq_io_data(args, data, x_offsets, y_offsets):
    num_samples, num_nodes, _ = data.shape
    data = data[:, :, 0:1]
    mu, sigma = np.mean(data), np.std(data)
    data = (data - mu) / sigma
    time_of_day = (np.arange(0, num_samples) % args.window)[:, np.newaxis]
    time_of_day = np.repeat(time_of_day, num_nodes, axis=1)[:, :, np.newaxis]
    day_of_week = ((np.arange(0, num_samples) // args.window + args.begin_dow) % 7)[:, np.newaxis]
    day_of_week = np.repeat(day_of_week, num_nodes, axis=1)[:, :, np.newaxis]
    data = np.concatenate([data, time_of_day, day_of_week], axis=2)
    x, y = [], []
    min_t = abs(min(x_offsets)) 
    max_t = abs(num_samples - abs(max(y_offsets))) 
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]  
        y_t = data[t + y_offsets, ...] 
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0) 
    y = np.stack(y, axis=0) 
    return x, y, mu, sigma


def generate_train_val_test(args):
    data_seq = np.load(args.traffic_df_filename)['data'][:, :, 0:1]
    r = generate_weight(args, data_seq[:,:,0])
    np.save(args.weight_dir + 'eval_mean_ratio.npy', r)
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(args.y_start, (seq_length_y + 1), 1)
    x, y, mu, sigma = generate_graph_seq2seq_io_data(args, data_seq, x_offsets=x_offsets, y_offsets=y_offsets)
    np.save(args.dataset_dir + "dist.npy", np.array([mu, sigma]))
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]
    print(f'y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}')
    train_val_raw_data = data_seq[:num_train + num_val + seq_length_y,:,0]
    r = generate_weight(args, train_val_raw_data)
    np.save(args.weight_dir + 'loss_mean_ratio.npy', r)
    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.dataset_dir, f"{cat}.npz"),
            x=_x,
            y=_y
        )


def generate_adj(N, dir, sigma):
    adj = np.zeros((N, N))
    df = pd.read_csv(f'{dir}/distance.csv')
    from_nodes = df['from']
    to_nodes = df['to']
    costs = df['cost']
    for frm, to, cost in zip(from_nodes, to_nodes, costs):
        adj[frm, to] = adj[to, frm] = cost

    preprocess_adj = np.zeros_like(adj)
    for i in range(N):
        for j in range(N):
            if adj[i][j] != 0.0:
                preprocess_adj[i][j] = np.exp(-adj[i][j]/(sigma * sigma))
    np.save(f'{dir}/graph/weighted.npy', preprocess_adj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./BJGRID/", help="folder dir")
    parser.add_argument('--traffic_df_filename', type=str, default="./BJGRID/bjgrid.npz", help="dataset dir")
    parser.add_argument('--seq_length_x', type=int, default=12, help='input sequence length')
    parser.add_argument('--seq_length_y', type=int, default=12, help='output sequence length')
    parser.add_argument('--y_start', type=int, default=1, help='day start')
    parser.add_argument('--seed_num', type=int, default=20, help='random seed')
    parser.add_argument('--window', type=int, default=288, help='time points per day')
    parser.add_argument('--node_nums', type=int, default=500, help='number of nodes')
    parser.add_argument('--sigma', type=float, default=3.0, help='graph initialize')
    parser.add_argument('--begin_dow', type=int, default=0, help='begin day of week')
    args = parser.parse_args()
    args.dataset_dir = args.dir + 'dataset/'
    args.graph_dir = args.dir + 'graph/'
    args.weight_dir = args.dir + 'weight/'
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)
    if not os.path.exists(args.graph_dir):
        os.makedirs(args.graph_dir)
    if not os.path.exists(args.weight_dir):
        os.makedirs(args.weight_dir)
    generate_train_val_test(args)
    generate_adj(args.node_nums, args.dir, args.sigma)