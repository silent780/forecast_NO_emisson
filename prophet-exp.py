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
one_data = raw_data["2024-01-12 00:00:00":"2024-01-12 23:59:59"]

test_data = one_data[[r"VA.NOX($mg/m^{3}$)"]]

# 转换列名
test_data.columns = ["y"]
test_data.loc[:, "ds"] = test_data.index
n = len(test_data)

train = test_data.iloc[: int(n * 0.8), :]
test = test_data.iloc[int(n * 0.8) :, :]

print(test_data.head())

m = Prophet()
m.fit(train)
fcst = m.predict(test)
fig = m.plot(fcst)
