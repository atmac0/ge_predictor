import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

df = pd.read_csv('data/rune_data.csv')
df.set_index('timestamp')
df.head()

print(df.describe())
print(df.shape)

ax = df.hist(bins=30)
#plt.show()

sns.boxplot(x=df['Chaos_rune'])

plt.show()

plt.figure(figsize=(20,10))
c = df.corr()
sns.heatmap(c,annot=True)

train_ratio = 0.8 # percent of dataset used for training



