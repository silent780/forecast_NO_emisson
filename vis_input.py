import pandas as pd
import ydata_profiling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

# 设置全局字体
matplotlib.rcParams["font.family"] = "SimSun"


def remove_parentheses(string):
    # 使用正则表达式匹配括号及其内部内容，并替换为空字符串
    pattern = r"\([^()]*\)"
    result = re.sub(pattern, "", string)
    return result


def remove_backslash(string):
    # 使用 replace 方法将反斜杠替换为空字符串
    result = string.replace("/", "")
    return result


raw_data = pd.read_csv(r"data\qiyeshuju-4S间隔.csv", encoding="gbk")
print(raw_data.head())
raw_data.index = pd.to_datetime(raw_data["时间"])
raw_data.drop(["时间"], axis=1, inplace=True)
raw_data.drop(["右侧换火信号"], axis=1, inplace=True)
columns = raw_data.columns
print(columns)
for column in columns:
    plt.figure(figsize=(100, 25))
    print(raw_data[column].head())

    raw_data[column].plot()
    # 设置保存路径和文件名
    save_path = "pic\\" + remove_backslash(rf"{column}.jpg")

    # 保存图表
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.title("折线图")
    plt.legend()
    plt.savefig(save_path)
    plt.close("all")
