import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from model.himoe import HiMoE
from metric import masked_mae_torch, masked_rmse_torch, masked_wae_torch, masked_saes_torch


def param_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model

        
def train(args):
    model = HiMoE(args=args).to(args.device)
    lossfunc = nn.L1Loss().to(args.device)
    if args.loss_function == 'masked_mae':
        lossfunc = masked_mae_torch
    elif args.loss_function == 'masked_rmse':
        lossfunc = masked_rmse_torch
    elif args.loss_function=='masked_rae_std':
        lossfunc = lambda x, y: masked_wae_torch(torch.Tensor(1 / args.loss_mean_ratio).reshape(1, -1, 1).to(args.device), x, y) + \
                                0.5 * masked_saes_torch(torch.Tensor(args.loss_mean_ratio).to(args.device), x, y)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_epoch = 0
    best_validation_loss = np.inf

    args.logger.info('Training start')
    model.train()
    for epoch in range(0, args.epoch):
        training_loss, cnt = 0.0, 0
        for batch_idx, batch_data in enumerate(args.train_loader):
            model_input, tod, dow, real_output = batch_data
            optimizer.zero_grad()
            pred = model(model_input) * args.sigma + args.mu
            loss = lossfunc(real_output, pred)
            training_loss += float(loss)
            loss.backward()
            optimizer.step()
            cnt += 1
        training_loss /= cnt
        
        validation_loss, cnt = 0.0, 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(args.val_loader):
                model_input, tod, dow, model_output = batch_data
                pred = model(model_input)* args.sigma + args.mu
                loss = lossfunc(model_output, pred)
                validation_loss += float(loss)
                cnt += 1
        validation_loss /= cnt

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")
        if validation_loss <= best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss
            torch.save({'model_state_dict': model.state_dict()}, f'{args.model_dir}{args.dataset}/{args.model_name}_{epoch}.pkl')

    args.logger.info(f'Best Epoch: {best_epoch}')
    args.best_model_path = f'{args.model_dir}{args.dataset}/{args.model_name}_{best_epoch}.pkl'