import pandas as pd

df = pd.read_csv("dataset/weather/full_weather.csv")

# 提取前1000行
top_1000 = df.head(1000)

# 保存到新的CSV文件
top_1000.to_csv('./dataset/weather/weather.csv', index=False)