import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from IPython.display import display, Markdown
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, LSTM
from neuralforecast.utils import AirPassengersDF

import pytorch_lightning as pl


from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

# 设置全局字体
matplotlib.rcParams["font.family"] = "SimSun"


# def ts_profiling(df: pd.DataFrame, output_file: str = "profiling_report.html") -> None:
#     profile = ydata_profiling.ProfileReport(df, tsmode=True)
#     profile.to_file(output_file)


raw_data = pd.read_csv(r"data\qiyeshuju-4S间隔.csv", encoding="gbk")
raw_data.index = pd.to_datetime(raw_data["时间"])
raw_data.drop("时间", axis=1, inplace=True)

columns = raw_data.columns
print(columns)

# 选择一天的数据
print(raw_data.index)
one_data = raw_data["2024-01-12 00:22:00":"2024-01-12 23:59:59"]

test_data = one_data[[r"VA.NOX($mg/m^{3}$)"]]

test_data.columns = ["y"]
test_data.loc[:, "ds"] = test_data.index
test_data.loc[:, "unique_id"] = [1.0 for i in range(len(one_data))]
test_data.head()


n = len(test_data)
train = test_data.iloc[: int(n * 0.8), :]
test = test_data.iloc[int(n * 0.8) :, :]
print(n)


# Split data and declare panel dataset
Y_df = AirPassengersDF
Y_train_df = train
Y_test_df = test


nf = NeuralForecast(
    models=[
        LSTM(
            h=12,
            input_size=-1,
            loss=DistributionLoss(distribution="Normal", level=[80, 90]),
            scaler_type="robust",
            encoder_n_layers=2,
            encoder_hidden_size=128,
            context_size=10,
            decoder_hidden_size=128,
            decoder_layers=2,
            max_steps=200,
            futr_exog_list=["y_[lag12]"],
            # hist_exog_list=['y_[lag12]'],
            stat_exog_list=["airline1"],
        )
    ],
    freq="4s",
)
nf.fit(df=Y_train_df)
Y_hat_df = nf.predict(futr_df=Y_test_df)

Y_hat_df = Y_hat_df.reset_index(drop=False).drop(columns=["unique_id", "ds"])
plot_df = pd.concat([Y_test_df, Y_hat_df], axis=1)
plot_df = pd.concat([Y_train_df, plot_df])

plot_df = plot_df[plot_df.unique_id == "Airline1"].drop("unique_id", axis=1)
plt.plot(plot_df["ds"], plot_df["y"], c="black", label="True")
plt.plot(plot_df["ds"], plot_df["LSTM"], c="purple", label="mean")
plt.plot(plot_df["ds"], plot_df["LSTM-median"], c="blue", label="median")
plt.fill_between(
    x=plot_df["ds"][-12:],
    y1=plot_df["LSTM-lo-90"][-12:].values,
    y2=plot_df["LSTM-hi-90"][-12:].values,
    alpha=0.4,
    label="level 90",
)
plt.legend()
plt.grid()
plt.plot()
plt.show()
