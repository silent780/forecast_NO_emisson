# 导入需要的包
"""
@File    :   ARIMA_exp.py
@Time    :   2024/01/29 15:59:47
@Author  :   glx 
@Version :   1.0
@Contact :   18095542g@connect.polyu.hk
@Desc    :   None
"""

# here put the import lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_absolute_error


import matplotlib

# # 设置全局字体
matplotlib.rcParams["font.family"] = "SimSun"

raw_data = pd.read_csv(r"data\qiyeshuju-4S间隔.csv", encoding="gbk")
print(raw_data.head())
raw_data.index = pd.to_datetime(raw_data["时间"])
raw_data.drop(["时间"], axis=1, inplace=True)
raw_data.drop(["右侧换火信号"], axis=1, inplace=True)
columns = raw_data.columns

# 划分训练集和测试集，假设用前80%的数据作为训练集，后20%的数据作为测试集

one_data = raw_data["2024-01-02 00:00:00":"2024-01-02 23:59:59"]
n = len(one_data)
test_data = one_data[[r"VA.NOX($mg/m^{3}$)"]]
# 转换列名
test_data.columns = ["y"]
test_data.loc[:, "ds"] = test_data.index


train = test_data.iloc[: int(n * 0.8), :]
test = test_data.iloc[int(n * 0.8) :, :]

# 检验数据的平稳性，使用ADF检验
print("ADF检验结果：")
adf_test = sm.tsa.stattools.adfuller(train["y"])
adf_output = pd.Series(adf_test[0:4], index=["检验统计量", "p值", "滞后阶数", "样本数"])
for key, value in adf_test[4].items():
    adf_output["%s水平的临界值" % key] = value
print(adf_output)

# 如果数据不平稳，进行差分，直到数据平稳
# 这里假设数据是平稳的，不需要差分
# 如果需要差分，可以使用以下代码
# diff = train.diff(1) # 一阶差分
# diff.dropna(inplace=True) # 去掉缺失值
# train = diff # 用差分后的数据作为训练集

# 检验数据的白噪声性，使用Ljung-Box检验
print("Ljung-Box检验结果：")
lb_test = acorr_ljungbox(train["y"], lags=[6, 12], boxpierce=True)
print(lb_test.head())
lb_output = pd.DataFrame(
    {
        "LB统计量": lb_test["lb_stat"],
        "LB p值": lb_test["lb_pvalue"],
        "BP统计量": lb_test["bp_stat"],
        "BP p值": lb_test["bp_pvalue"],
    },
    index=["滞后6阶", "滞后12阶"],
)
print(lb_output)

# 如果数据不是白噪声，计算ACF和PACF，选择合适的p和q参数
# 这里假设数据是非白噪声的，需要选择p和q
# 如果数据是白噪声，无法用arima模型进行预测
# 计算ACF和PACF，并绘制图形
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train["y"], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train["y"], lags=40, ax=ax2)
plt.show()

# 根据ACF和PACF的结果，选择合适的p和q
# 这里假设选择的p=2，q=1
# 也可以用AIC或BIC准则来选择最优的p和q，例如：
order_select = sm.tsa.arma_order_select_ic(
    train, ic=["aic", "bic"], trend="n", max_ar=5, max_ma=5
)
print("AIC准则下的最优阶数：", order_select.aic_min_order)
print("BIC准则下的最优阶数：", order_select.bic_min_order)

# 将 train 转换为 NumPy 数组
train_array = np.asarray(train)[:, 0]
train_array = np.nan_to_num(train_array)  # 清除缺失值

print(train_array.dtype)
# 使用转换后的数据进行 ARIMA 模型拟合
model = sm.tsa.ARIMA(
    train_array, order=(order_select.aic_min_order, 0, order_select.bic_min_order)
)


results = model.fit()
print("模型参数：")
print(results.params)

# 进行预测，返回预测值，标准误差，置信区间
preds = results.forecast(len(test))
pred_values = preds[0]  # 预测值
pred_se = preds[1]  # 标准误差
pred_ci = preds[2]  # 置信区间

# 绘制预测结果图
plt.plot(test.index, test["VA.NOX($mg/m^{3}$)"], label="真实值")
plt.plot(test.index, pred_values, label="预测值")
plt.fill_between(test.index, pred_ci[:, 0], pred_ci[:, 1], color="pink", label="置信区间")
plt.legend()
plt.title("预测结果图")
plt.show()

# 评估模型效果，使用平均绝对误差（MAE）作为评价指标
mae = mean_absolute_error(test["VA.NOX($mg/m^{3}$)"], pred_values)
print("平均绝对误差：", mae)

# 残差分析，检验残差的正态性和随机性
resid = test["VA.NOX($mg/m^{3}$)"] - pred_values  # 残差
# 绘制残差图
plt.plot(test.index, resid)
plt.title("残差图")
plt.show()
# 计算残差的均值和标准差
resid_mean = np.mean(resid)
resid_std = np.std(resid)
print("残差的均值：", resid_mean)
print("残差的标准差：", resid_std)
# 绘制残差的直方图和QQ图，检验正态性
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
resid.hist(ax=ax1, bins=20)
ax1.set_title("残差直方图")
ax2 = fig.add_subplot(212)
sm.qqplot(resid, line="q", ax=ax2)
ax2.set_title("残差QQ图")
plt.show()
# 使用Ljung-Box检验残差的随机性
lb_test2 = acorr_ljungbox(resid, lags=[6, 12], boxpierce=True)
lb_output2 = pd.DataFrame(
    {
        "LB统计量": lb_test2[0],
        "LB p值": lb_test2[1],
        "BP统计量": lb_test2[2],
        "BP p值": lb_test2[3],
    },
    index=["滞后6阶", "滞后12阶"],
)
print("残差的Ljung-Box检验结果：")
print(lb_output2)
