import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
# print(df.head())
trian=df[0:10392]
test=df[10392:]
# print(test)
df["Timestamp"]=pd.to_datetime(df["Datetime"],format="%d-%m-%Y %H:%M")

df.index=df["Timestamp"]
# print(df.head())
df=df.resample("D").mean()
# print(df.head())

trian["Timestamp"]=pd.to_datetime(trian["Datetime"],format="%d-%m-%Y %H:%M")

trian.index=trian["Timestamp"]
# print(trian.head())
trian=trian.resample("D").mean()
# print(trian.head())

test['Timestamp'] = pd.to_datetime(test['Datetime'], format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
# print(test.head())
test = test.resample('D').mean()
# print(test.head())

# trian.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
# test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
# plt.show()

#朴素法预测
#如果数据集在一段时间内都很稳定，我们想预测第二天的价格，
# 可以取前面一天的价格，预测第二天的值。
# 这种假设第一个预测点和上一个观察点相等的预测方法就叫朴素法
dd=np.asarray(trian["Count"])
y_hat=test.copy()
y_hat["native"]=dd[len(dd)-1]
# print(y_hat)
plt.figure(figsize=(12, 8))

# print(y_hat.index)
# plt.plot(trian.index,trian["Count"], label='Train')
# plt.plot(test.index, test['Count'], label='Test')
# plt.plot(y_hat.index, y_hat['native'], label='Naive Forecast')
# plt.legend(loc='best')
# plt.title("Naive Forecast")
# plt.show()
from sklearn.metrics import mean_squared_error
from math import sqrt
rms=sqrt(mean_squared_error(test["Count"],y_hat["native"]))
print(rms)

#简单平均法












