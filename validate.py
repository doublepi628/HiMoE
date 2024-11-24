import numpy as np
import torch, torch.nn as nn
from model.himoe import HiMoE
from metric import masked_mae_np, masked_mape_np, masked_rmse_np, masked_wae_np, masked_saes_np

def validate(args):
    model =  HiMoE(args=args).to(args.device)
    model.load_state_dict(torch.load(args.best_model_path, args.device)["model_state_dict"])

    pred_list, truth_list = [], []
    args.logger.info('Validation start')
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(args.val_loader):
            model_input, tod, dow, truth = batch_data
            pred = model(model_input) * args.sigma + args.mu
            pred_list.append(pred.cpu().data.numpy())
            truth_list.append(truth.cpu().data.numpy())
    pred_np = np.concatenate(pred_list, 0)
    truth_np = np.concatenate(truth_list, 0)
    
    args.logger.info("Best Model:")
    for horizon in [3, 6, 12]:
        pred_horizon, truth_horizon = np.expand_dims(pred_np[:,:,horizon-1], axis=-1), np.expand_dims(truth_np[:,:,horizon-1], axis=-1)
        mae, mape, rmse = masked_mae_np(truth_horizon, pred_horizon, 0), masked_mape_np(truth_horizon, pred_horizon, 0), masked_rmse_np(truth_horizon, pred_horizon, 0)
        wae = masked_wae_np(1/args.eval_mean_ratio, truth_horizon, pred_horizon, 0)
        sase = masked_saes_np(args.eval_mean_ratio, truth_horizon, pred_horizon, 0)
        args.logger.info("[T = {:2d}] MAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}\tWAE\t{:.4f}\tSARE\t{:.4f}".format(horizon,mae,rmse,mape,wae,sase))
    mae, mape, rmse = masked_mae_np(truth_np, pred_np, 0), masked_mape_np(truth_np, pred_np, 0), masked_rmse_np(truth_np, pred_np, 0)
    wae = masked_wae_np(1/args.eval_mean_ratio, truth_np, pred_np, 0)
    sase = masked_saes_np(args.eval_mean_ratio, truth_np, pred_np, 0)
    args.logger.info("[ MEAN ] MAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}\tWAE\t{:.4f}\tSARE\t{:.4f}".format(mae,rmse,mape,wae,sase))
