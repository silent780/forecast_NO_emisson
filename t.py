# here put the import lib

import pandas as pd
import ydata_profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
# for column in columns:
#     # print(raw_data[column].head())
#     try:
#         raw_data[column].plot()
#         # 设置保存路径和文件名
#         save_path = rf"{column}_plot.png"

#         # 保存图表
#         plt.xlabel("日期")
#         plt.ylabel("值")
#         plt.title("折线图")
#         plt.legend()
#         plt.savefig(save_path)
#         plt.close("all")
#     except:
#         pass
