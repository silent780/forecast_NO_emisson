import pandas as pd
import numpy as np
import random
import os
import joblib
import time
from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cpu")  # 使用CPU训练
dtype = torch.float64
torch.set_default_dtype(dtype)

if not os.path.exists("tmp"):
    os.makedirs("tmp")
if not os.path.exists("model"):
    os.makedirs("model")

DEFAULT_RANDOM_SEED = 42


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        pass


def scale_train_dataset():
    """将训练集中的5个指标归一化为均值为0、方差为1的分布，保存归一化结果和StandardScaler"""
    train = pd.read_csv("train.csv", parse_dates=["time"])
    scaler = StandardScaler()
    train_transformed = train.copy()
    train_transformed[["dv1", "dv2", "mv1", "cv1", "cv2"]] = scaler.fit_transform(
        train_transformed[["dv1", "dv2", "mv1", "cv1", "cv2"]]
    )
    train_transformed.to_pickle("tmp/train_transformeed.pkl")
    joblib.dump(scaler, f"tmp/scaler.pkl")


def save_model(model, n_input, n_hidden, seed):
    torch.save(model.state_dict(), f"model/model{n_input}_{n_hidden}_{seed}.pth")


class CustomNARX(nn.Module):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_targets=2,
        n_exogs=3,
        device=device,
        dtype=dtype,
    ):
        """
        自定义NARX模型。
        n_input: 模型每次输入的历史时间点数
        n_output: 模型在训练过程中每次预测的时间点数
        n_hidden: 模型从每个字段中提取几个非线性特征
        n_targets: 模型预测的目标字段数，本题目中为2
        n_exogs: 模型输入的外部变量数，本题目中为3
        """
        super().__init__()
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.n_input, self.n_output = n_input, n_output
        self.n_targets, self.n_exogs = n_targets, n_exogs
        self.n_series = n_targets + n_exogs
        self.linear1 = nn.Linear(
            n_input * (self.n_series), n_targets, bias=True, **self.factory_kwargs
        )  # 输入到输出的线性变换参数W1、B1
        self.linear2 = nn.ModuleList(
            [
                nn.Linear(n_input, n_hidden, **self.factory_kwargs)
                for i in range(self.n_series)
            ]
        )  # 输入到非线性特征的线性变换参数W21-W25、B21-B25
        self.linear3 = nn.Linear(
            n_hidden * (self.n_series), n_targets, bias=True, **self.factory_kwargs
        )  # 非线性特征到输出的线性变换参数W3、B3
        self.sigmoid = nn.Sigmoid()  # sigmoid非线性变换
        self.flatten = torch.nn.Flatten()

    def single_predict(self, target, exog):
        """
        使用历史数值进行单个时间点的预测。
        target和exog的shape均应为(batch_size, n_input，字段数)
        """
        target_exog = torch.cat([target, exog], dim=2)
        output_1 = self.linear1(self.flatten(target_exog))
        output_2 = self.linear3(
            self.sigmoid(
                torch.cat(
                    [
                        self.linear2[i](self.flatten(target_exog[:, :, i]))
                        for i in range(self.n_series)
                    ],
                    dim=1,
                )
            )
        )
        return output_1 + output_2 + target[:, -1, :]

    def forward(self, target, exog):
        """
        使用历史数值和相应时间内的外部变量连续预测n_output个时间点的值。
        target的shape为(batch_size, n_input，字段数)，exog的shape为(batch_size, n_input + n_output，字段数)
        """
        extended_target = torch.cat(
            [
                target,
                torch.empty(
                    (target.shape[0], self.n_output, self.n_targets),
                    **self.factory_kwargs,
                ),
            ],
            dim=1,
        )
        for i in range(self.n_output):
            extended_target[:, i + self.n_input, :] = self.single_predict(
                extended_target[:, i : i + self.n_input, :],
                exog[:, i : i + self.n_input, :],
            )
        return extended_target[:, -self.n_output :, :]


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        target: np.array,
        exog: np.array,
        n_input: int,
        n_output: int,
        device=device,
        dtype=dtype,
    ):
        """
        用于训练CustomNARX模型的Dataset。
        target: 待预测字段，shape为(时间点数, 字段数)
        exog: 外部变量字段，shape为(时间点数, 字段数)
        n_input: 模型每次输入的历史时间点数
        n_output: 模型在训练过程中每次预测的时间点数
        """
        super().__init__()
        assert target.shape[0] == exog.shape[0]
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.target = torch.tensor(target, **self.factory_kwargs)
        self.exog = torch.tensor(exog, **self.factory_kwargs)
        self.n_input, self.n_output = n_input, n_output

    def __len__(self):
        return self.target.shape[0] - self.n_input - self.n_output + 1

    def __getitem__(self, idx):
        """
        每次取n_input个时间点的target值和对应的(n_input + n_output)个时间点的exog值用于模型输入，对应的n_output个时间点的target值作为模型输出
        """
        return (
            self.target[idx : idx + self.n_input, :],
            self.exog[idx : idx + self.n_input + self.n_output, :],
            self.target[idx + self.n_input : idx + self.n_input + self.n_output, :],
        )


def train(dataloader, model, loss_fn, optimizer):
    """训练一个epoch。"""
    model.train()
    losses = []
    for batch, (target, exog, y) in enumerate(dataloader):
        pred = model(target, exog)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.array(losses), np.mean(losses), np.median(losses)


def train_a_model(
    n_input, n_output, n_hidden, n_epochs, batch_size, lr_init, seed=DEFAULT_RANDOM_SEED
):
    """训练一个模型并返回。"""
    seedBasic(seed)
    seedTorch(seed)  # 设定随机种子来确保可复现性
    train_transformed = pd.read_pickle("tmp/train_transformeed.pkl")
    dataset = TimeSeriesDataset(
        train_transformed[["cv1", "cv2"]].values,
        train_transformed[["dv1", "dv2", "mv1"]].values,
        n_input=n_input,
        n_output=n_output,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.MSELoss()
    print_freq = n_epochs // 32
    print(f"n_input={n_input}, n_output={n_output}, n_hidden={n_hidden}")
    model = CustomNARX(n_input=n_input, n_output=n_output, n_hidden=n_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=n_epochs
    )  # 线性学习率衰减
    start_time = time.time()
    for epoch in tqdm(range(n_epochs)):
        losses, mean_loss, median_loss = train(dataloader, model, loss_fn, optimizer)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        if epoch == 0 or (epoch + 1) % print_freq == 0:
            print(
                f"epoch {epoch}\tmean loss: {mean_loss}\tcurrent learning rate: {current_lr}"
            )
    print(f"time used: {time.time() - start_time}s")
    return model


def quick_sort(array):
    if len(array) < 2:
        return array
    else:
        pivot = array[0]
        less = [i for i in array[1:] if i <= pivot]  # 由所有小于等于基准值的元素组成的子数组
        greater = [i for i in array[1:] if i > pivot]  # 由所有大于基准值的元素组成的子数组
        return quick_sort(less) + [pivot] + quick_sort(greater)
