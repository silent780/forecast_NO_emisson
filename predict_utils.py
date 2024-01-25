import pandas as pd
import numpy as np
import os
import joblib
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
device = torch.device('cpu')
dtype = torch.float64
torch.set_default_dtype(dtype)
import train_utils

if not os.path.exists('submit'):
    os.makedirs('submit')
    
scaler = joblib.load('tmp/scaler.pkl')

def scale_predict_dataset():
    '''将预测输入数据拆分成3组，并将其中的5个指标归一化为均值为0、方差为1的分布，保存归一化结果'''
    predict_B = pd.read_csv('predict_B.csv', skiprows=1, encoding='GBK')
    ret = predict_B.iloc[:, :6], predict_B.iloc[:, 7:13], predict_B.iloc[:, 14:20]
    ret[1].columns , ret[2].columns = ret[0].columns, ret[0].columns
    for i, df in enumerate(ret):
        new_time = df.time.copy()
        new_time[0:1] = '2023-01-01 ' + new_time[0:1]
        new_time[1:] = '2023-01-02 ' + new_time[1:]
        df.time = pd.to_datetime(new_time)
        df[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']] = scaler.transform(df[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']])
        df.to_pickle(f'tmp/predict_B_{i}.pkl')
        
def load_model(n_input, n_output, n_hidden, seed):
    model = train_utils.CustomNARX(n_input=n_input, n_output=n_output, n_hidden=n_hidden)
    model.load_state_dict(torch.load(f'model/model{n_input}_{n_hidden}_{seed}.pth'))
    return model

def modify_by_end(series: np.array, end):
    '''
    根据一段序列之后的一个值来修正这段序列的值。
    '''
    end_ratio = np.linspace(0, 1, num=len(series) + 2)[1:-1]
    return series * (1. - end_ratio) + end * end_ratio

def predict(predict: pd.DataFrame, model: train_utils.CustomNARX, n_input: int, filename=''):
    '''进行一组数据的预测'''
    append_before_predict_range = pd.DataFrame({
        'time': predict.time[:n_input - 1] - (n_input - 1) * pd.Timedelta(seconds=30),
        'dv1': predict.loc[0, 'dv1'],
        'dv2': predict.loc[0, 'dv2'],
        'mv1': predict.loc[0, 'mv1'],
        'cv1': predict.loc[0, 'cv1'],
        'cv2': predict.loc[0, 'cv2']
    })
    predict = pd.concat([append_before_predict_range, predict], ignore_index=True) # 将predict延拓到之前的n_input - 1个时间点，这些时间点上的各指标数值均取初始值
    original_predict = predict.copy() # 复制包含空值的初始predict，用于画图
    modified_predict_for_model_input = predict.copy() # modified_predict_for_model_input里的数值会根据离散采样的真实cv1和cv2值修正，修正后的数值仅用于模型输入，不参与最终的输出
    
    # 确定当前predict dataframe中有空值的行的行号，再从中确定连续空值段的起始和结束行号
    nan_rows = list(predict[np.isnan(predict).any(axis=1)].index.values)
    nan_periods = []
    nan_period_start = nan_rows[0]
    for i in range(len(nan_rows) - 1):
        if nan_rows[i + 1] - nan_rows[i] == 1:
            continue
        else:
            nan_periods.append((nan_period_start, nan_rows[i]))
            nan_period_start = nan_rows[i + 1]
    nan_periods.append((nan_period_start, nan_rows[-1]))
    
    # 对每一个连续的空值段，对其中的每个时间点逐个用历史数值进行预测。
    for nan_period_start, nan_period_end in tqdm(nan_periods):
        for i in range(nan_period_start, nan_period_end + 1):
            target, exog = modified_predict_for_model_input.loc[i-n_input:i-1, ['cv1', 'cv2']].values, modified_predict_for_model_input.loc[i-n_input:i-1, ['dv1', 'dv2', 'mv1']].values
            target, exog = np.stack((target,)), np.stack((exog,))
            target, exog = torch.tensor(target, device='cpu', dtype=torch.float64), torch.tensor(exog, device='cpu', dtype=torch.float64)
            new_prediction = model.single_predict(target, exog).detach().numpy()[0] # 调用模型的单点预测函数
            predict.loc[i, ['cv1', 'cv2']] = new_prediction # 将预测结果填入predict中
            modified_predict_for_model_input.loc[i, ['cv1', 'cv2']] = new_prediction # 将预测结果填入modified_predict_for_model_input中
        if nan_period_end == predict.index[-1]:
            break
        # 在每个连续的空值段结束时，根据下一个时间点的真实cv1和cv2修正modified_predict_for_model_input。根据比赛规则限制，最终输出的predict不会被修正。
        for kpi in ['cv1', 'cv2']:
            modified_predict_for_model_input.loc[nan_period_start:nan_period_end, kpi] = modify_by_end(modified_predict_for_model_input.loc[nan_period_start:nan_period_end, kpi], modified_predict_for_model_input.loc[nan_period_end + 1, kpi])

    # 将归一化的结果还原成原始数值
    predict[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']] = scaler.inverse_transform(predict[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']])
    original_predict[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']] = scaler.inverse_transform(original_predict[['dv1', 'dv2', 'mv1', 'cv1', 'cv2']])
    
    # 将predict的值绘制为较小的红点，将original_predict的值绘制为较大的蓝点，以观察预测效果。
    for kpi in ['cv1', 'cv2']:
        _ = plt.figure(figsize=(10, 2))
        _ = plt.title(f'{kpi}')
        _ = plt.scatter(predict.time, predict[kpi], s=1, c='r')
        _ = plt.scatter(original_predict.time, original_predict[kpi], s=5, c='b')
        _ = plt.savefig(f'figures/{filename}_{kpi}.svg')
        _ = plt.show()
        
    return predict