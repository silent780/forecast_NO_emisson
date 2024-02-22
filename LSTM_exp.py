# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib

# from IPython.display import display, Markdown
# from neuralforecast import NeuralForecast
# from neuralforecast.models import NBEATS, NHITS, LSTM
# from neuralforecast.utils import AirPassengersDF

# import pytorch_lightning as pl


# from neuralforecast import NeuralForecast
# from neuralforecast.models import LSTM
# from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
# from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
# from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

# # 设置全局字体
# matplotlib.rcParams["font.family"] = "SimSun"


# # def ts_profiling(df: pd.DataFrame, output_file: str = "profiling_report.html") -> None:
# #     profile = ydata_profiling.ProfileReport(df, tsmode=True)
# #     profile.to_file(output_file)


# raw_data = pd.read_csv(r"data\qiyeshuju-4S间隔.csv", encoding="gbk")
# raw_data.index = pd.to_datetime(raw_data["时间"])
# raw_data.drop("时间", axis=1, inplace=True)

# columns = raw_data.columns
# print(columns)

# # 选择一天的数据
# print(raw_data.index)
# one_data = raw_data["2024-01-12 00:22:00":"2024-01-12 23:59:59"]

# test_data = one_data[[r"VA.NOX($mg/m^{3}$)"]]

# test_data.columns = ["y"]
# test_data.loc[:, "ds"] = test_data.index
# test_data.loc[:, "unique_id"] = [1.0 for i in range(len(one_data))]
# test_data.head()


# n = len(test_data)
# train = test_data.iloc[: int(n * 0.8), :]
# test = test_data.iloc[int(n * 0.8) :, :]
# print(n)


# # Split data and declare panel dataset
# Y_df = AirPassengersDF
# Y_train_df = train
# Y_test_df = test


# nf = NeuralForecast(
#     models=[
#         LSTM(
#             h=12,
#             input_size=-1,
#             loss=DistributionLoss(distribution="Normal", level=[80, 90]),
#             scaler_type="robust",
#             encoder_n_layers=2,
#             encoder_hidden_size=128,
#             context_size=10,
#             decoder_hidden_size=128,
#             decoder_layers=2,
#             max_steps=200,
#             futr_exog_list=["y_[lag12]"],
#             # hist_exog_list=['y_[lag12]'],
#             stat_exog_list=["airline1"],
#         )
#     ],
#     freq="4s",
# )
# nf.fit(df=Y_train_df)
# Y_hat_df = nf.predict(futr_df=Y_test_df)

# Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=["unique_id", "ds"])
# plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
# plot_df = pd.concat([Y_train_df, plot_df])

# plot_df = plot_df[plot_df.unique_id == "Airline1"].drop("unique_id", axis=1)
# plt.plot(plot_df["ds"], plot_df["y"], c="black", label="True")
# plt.plot(plot_df["ds"], plot_df["LSTM"], c="purple", label="mean")
# plt.plot(plot_df["ds"], plot_df["LSTM-median"], c="blue", label="median")
# plt.fill_between(
#     x=plot_df["ds"][-12:],
#     y1=plot_df["LSTM-lo-90"][-12:].values,
#     y2=plot_df["LSTM-hi-90"][-12:].values,
#     alpha=0.4,
#     label="level 90",
# )
# plt.legend()
# plt.grid()
# plt.plot()
# plt.show()


# ==========================================================================
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载和预处理数据
data = pd.read_csv("nox_onde_day.csv")
data = data["VA.NOX($mg/m^{3}$)"].values
data = data.astype("float32")
data = np.reshape(data, (-1, 1))

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

# 划分训练集和测试集
train_size = int(len(data) * 0.7)
train_set = data[:train_size]
test_set = data[train_size:]

# 转换为tensor
train_set = torch.FloatTensor(train_set).to(device)
test_set = torch.FloatTensor(test_set).to(device)


# 2. 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size).to(device),
            torch.zeros(1, 1, self.hidden_layer_size).to(device),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 3. 定义损失函数和优化器
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 150

for i in range(epochs):
    for seq in train_set:
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size).to(device),
            torch.zeros(1, 1, model.hidden_layer_size).to(device),
        )

        y_pred = model(seq)

        single_loss = loss_function(y_pred, seq)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")

# 保存模型
torch.save(model.state_dict(), "model.pth")

# 5. 进行预测
model.eval()

predictions = []
for i in range(5):  # 预测未来5分钟
    seq = torch.FloatTensor(test_set[-300:]).to(device)  # 使用最后5分钟的数据进行预测
    with torch.no_grad():
        model.hidden = (
            torch.zeros(1, 1, model.hidden_layer_size).to(device),
            torch.zeros(1, 1, model.hidden_layer_size).to(device),
        )
        predictions.append(model(seq).item())
        test_set = np.append(test_set.cpu(), model(seq).cpu())

# 可视化结果
plt.plot(range(len(predictions)), predictions, label="Predicted")
plt.legend()
plt.show()
