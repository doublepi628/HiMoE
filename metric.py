import numpy as np
import torch


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'),)
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)


def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = ((y_pred- y_true)**2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    idx = y_true > 1e-5
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.subtract(y_pred, y_true).astype('float32')) / np.abs(y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100
    

def masked_wae_np(weight, y_true, y_pred, null_val=np.nan):
    weight = weight.reshape(1, -1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask = mask * weight
        mask /= np.sum(mask)
        ae = np.abs(np.subtract(y_pred, y_true).astype('float32'),)
        wae = np.nan_to_num(mask * ae)
        return np.sum(wae)


def masked_rwse_np(weight, y_true, y_pred, null_val=np.nan):
    weight = weight.reshape(1, -1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask = mask * weight
        mask /= np.sum(mask)
        se = ((y_pred- y_true)**2)
        mse = np.nan_to_num(mask * se)
        return np.sqrt(np.sum(mse))


def masked_wape_np(weight, y_true, y_pred, null_val=np.nan):
    weight = weight.reshape(1, -1, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask = mask * weight
        mask /= np.sum(mask)
        ape = np.abs(np.subtract(y_pred, y_true).astype('float32')) / np.abs(y_true)
        mape = np.nan_to_num(mask * ape)
        return np.sum(mape) * 100


def masked_saes_np(ratio, y_true, y_pred, null_val=np.nan):
    aes = np.abs(y_true - y_pred) / np.expand_dims(ratio, axis=(0, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(aes)
        else:
            mask = np.not_equal(aes, null_val)
        return np.std(aes[mask])


def masked_mae_torch(labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_sase_np(ratio, y_true, y_pred, null_val=np.nan):
    aes = np.mean(np.abs(y_true - y_pred), axis=(0, 2)) / ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(aes)
        else:
            mask = np.not_equal(aes, null_val)
        return np.std(aes[mask])
    
def masked_saes_torch(ratio, labels, preds, null_val=np.nan):
    aes = torch.abs(preds - labels) / ratio.reshape((1, -1, 1))
    if np.isnan(null_val):
        mask = ~torch.isnan(aes)
    else:
        mask = (aes != null_val)
    return torch.std(aes * mask)


def masked_mse_torch(labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(labels, preds, null_val=np.nan):
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels, null_val=null_val))


def masked_mape_torch(labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels) / labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_wae_torch(weight, labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float() * weight
    mask /= torch.sum(mask)
    ae = torch.abs(preds- labels)
    wae = torch.nan_to_num(mask * ae)
    return torch.sum(wae)


def masked_rwse_torch(weight, labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask * weight
    mask /= torch.sum(mask)
    se = ((preds- labels)**2)
    mse = torch.nan_to_num(mask * se)
    return torch.sqrt(torch.sum(mse)).item()


def masked_wape_torch(weight, labels, preds, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask = mask * weight
    mask /= torch.sum(mask)
    ae = torch.abs((preds - labels) / labels)
    mae = torch.nan_to_num(mask * ae)
    return torch.sum(mae).item() * 100