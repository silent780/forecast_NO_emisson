# # here put the import lib


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
import matplotlib

# 设置全局字体
matplotlib.rcParams["font.family"] = "SimSun"

raw_data = pd.read_csv(r"data\qiyeshuju-4S间隔.csv", encoding="gbk")
raw_data.index = pd.to_datetime(raw_data["时间"])
raw_data.drop("时间", axis=1, inplace=True)

columns = raw_data.columns
print(columns)


from prophet import Prophet

# 选择一天的数据
print(raw_data.index)
one_data = raw_data["2024-01-02 00:00:00":"2024-01-02 23:59:59"]


test_data = one_data[[r"VA.NOX($mg/m^{3}$)"]]
test_data.columns = ["y"]
test_data.loc[:, "ds"] = test_data.index

# 测试间隔：
test_data["diff"] = test_data["ds"].diff()
if test_data["diff"].nunique() == 1:
    print("The time series data is equally spaced.")
else:
    print("The time series data is not equally spaced.")


# 要用后75个数据作为测试集，前面的作为训练集
train_data = test_data.iloc[:-75]
test_data = test_data.iloc[-75:]


m = Prophet(stan_backend="CMDSTANPY", growth="logistic", seasonality_mode="additive")
m.fit(test_data)

# future = m.make_future_dataframe(
#     periods=5,
#     freq="min",
# )
# future.tail()

# 使用测试集的时间戳进行预测
forecast = m.predict(test_data[["ds"]])

forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()
fig1 = m.plot_components(forecast)
# fig1.savefig("prophet.png")

# 绘制测试集的真实值和预测值
plt.plot(test_data["ds"], test_data["y"], label="真实值")
plt.plot(test_data["ds"], forecast["yhat"], label="预测值")
plt.plot(test_data["ds"], forecast["yhat_lower"], label="预测值下界")
plt.plot(test_data["ds"], forecast["yhat_upper"], label="预测值上界")
plt.legend()

plt.savefig("prophet2.png")
plt.show()


# 计算MAE和MAPE
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

mae = mean_absolute_error(test_data["y"], forecast["yhat"])
mape = mean_absolute_percentage_error(test_data["y"], forecast["yhat"])
print("MAE:", mae)
print("MAPE:", mape)
