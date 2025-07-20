import pandas as pd

df = pd.read_csv('data/train.csv')

task = df["answer"][36]
print(task)
