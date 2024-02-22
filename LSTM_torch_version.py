import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. 加载和预处理数据
data = pd.read_csv(r"data\nox_onde_day.csv")
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
train_set = torch.FloatTensor(train_set)
test_set = torch.FloatTensor(test_set)


# 2. 定义 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 3. 定义损失函数和优化器
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
epochs = 50

for i in range(epochs):
    for seq in train_set:
        optimizer.zero_grad()
        model.hidden_cell = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )

        y_pred = model(seq)

        single_loss = loss_function(y_pred, seq)
        single_loss.backward()
        optimizer.step()

    if i % 25 == 1:
        print(f"epoch: {i:3} loss: {single_loss.item():10.8f}")

print(f"epoch: {i:3} loss: {single_loss.item():10.10f}")
torch.save(model.state_dict(), "model.pth")
# 5. 进行预测
model.eval()

predictions = []
for i in range(5):  # 预测未来5分钟
    seq = torch.FloatTensor(test_set[-300:])  # 使用最后5分钟的数据进行预测
    with torch.no_grad():
        model.hidden = (
            torch.zeros(1, 1, model.hidden_layer_size),
            torch.zeros(1, 1, model.hidden_layer_size),
        )
        predictions.append(model(seq).item())
        test_set = np.append(test_set, model(seq))


# 可视化结果
plt.plot(range(len(predictions)), predictions, label="Predicted")
plt.legend()
plt.show()
