import numpy as np
import pandas as pd

mape = np.load('./mape.npy')
y_true = np.load('./y_true.npy')
y_pred = np.load('./y_pred.npy')

indices = np.where(mape > 100)
y_true_high_mape = y_true[indices]
y_pred_high_mape = y_pred[indices]

idx = y_true > 1e-5
y_true = y_true[idx]
y_pred = y_pred[idx]

mape = np.abs(np.subtract(y_pred, y_true).astype('float32')) / np.abs(y_true)
print(np.mean(mape))