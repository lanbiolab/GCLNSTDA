import numpy as np

dataset = np.array([0.4947
,0.4947
,0.4948
,0.4947
,0.4948
,0.4947
,0.4947
,0.4947
,0.4947
,0.4948
])
sd = np.std(dataset)
# 计算均值
mean_value = sum(dataset) / len(dataset)

# 打印均值
print("均值是:", mean_value)
print("Population standard deviation of the dataset is", sd)