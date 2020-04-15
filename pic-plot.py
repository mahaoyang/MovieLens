import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('C:/Users/99263/Desktop/1.csv')
d['cvr'] = d.apply(lambda x: x.buyuv / x['uv'], axis=1)
d['date'] = pd.to_datetime(d['date'], format='%Y%m%d')
date = d['date'].values
gmv = d['money'].values
cvr = d['cvr'].values
plt.plot(date, cvr)
plt.show()
