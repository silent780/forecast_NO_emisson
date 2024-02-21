# import numpy as np
# import pandas as pd
# from IPython.display import display, Markdown

# import matplotlib.pyplot as plt
# from neuralforecast import NeuralForecast
# from neuralforecast.models import NBEATS, NHITS
# from neuralforecast.utils import AirPassengersDF

# # Split data and declare panel dataset
# Y_df = AirPassengersDF
# Y_train_df = Y_df[Y_df.ds <= "1959-12-31"]  # 132 train
# Y_test_df = Y_df[Y_df.ds > "1959-12-31"]  # 12 test

# # Fit and predict with NBEATS and NHITS models
# horizon = len(Y_test_df)
# models = [
#     NBEATS(input_size=2 * horizon, h=horizon, max_steps=50),
#     NHITS(input_size=2 * horizon, h=horizon, max_steps=50),
# ]
# nf = NeuralForecast(models=models, freq="M")
# nf.fit(df=Y_train_df)
# Y_hat_df = nf.predict().reset_index()

# # Plot predictions
# fig, ax = plt.subplots(1, 1, figsize=(20, 7))
# Y_hat_df = Y_test_df.merge(Y_hat_df, how="left", on=["unique_id", "ds"])
# plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index("ds")

# plot_df[["y", "NBEATS", "NHITS"]].plot(ax=ax, linewidth=2)

# ax.set_title("AirPassengers Forecast", fontsize=22)
# ax.set_ylabel("Monthly Passengers", fontsize=20)
# ax.set_xlabel("Timestamp [t]", fontsize=20)
# ax.legend(prop={"size": 15})
# ax.grid()

# plt.show()
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MQLoss, DistributionLoss
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from neuralforecast.tsdataset import TimeSeriesDataset, TimeSeriesLoader

Y_train_df = AirPassengersPanel[
    AirPassengersPanel.ds < AirPassengersPanel["ds"].values[-12]
]  # 132 train
Y_test_df = AirPassengersPanel[
    AirPassengersPanel.ds >= AirPassengersPanel["ds"].values[-12]
].reset_index(
    drop=True
)  # 12 test

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
    freq="M",
)
nf.fit(df=Y_train_df, static_df=AirPassengersStatic)
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
