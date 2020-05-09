# import pandas as pd
#
# a = [{'a': 1, 'b': 2}, {'a': 1, 'b': 4}, {'a': 5, 'b': 6}]
# a = pd.DataFrame(a)
# a = a.groupby(['a'])
# for i in a:
#     print(i[1])

# from hdfs import InsecureClient
#
# client = InsecureClient('http://172.16.59.18:14000', user='root')
# client.makedirs('/temp')
#
# print([i for i in client.walk('/')])

# import pyhdfs
#
# client = pyhdfs.HdfsClient(hosts="172.16.59.18,9000", user_name="root")
# print(client.get_home_directory())
# print(client.get_active_namenode())
# print(client.listdir("/"))
# client.copy_from_local("C:/Users/99263/PycharmProjects/MovieLens/test.txt", "/user/hadoop/test.txt")
# # 打开一个远程节点上的文件，返回一个HTTPResponse对象
# response = client.open("/user/hadoop/test.txt")
# # 查看文件内容
# response.read()

import pandas as pd
from datetime import datetime

#
# d = [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 1, 'b': 2}, ]
# d = pd.DataFrame(d)
# d = d.reset_index()
# print(d)

# da = pd.read_csv('C:/Users/99263/Desktop/1.csv')
# da['start_time'] = pd.to_datetime(da['start_time'])
# da['end_time'] = pd.to_datetime(da['end_time'])
#
# d = da[da['start_time'] < datetime(2020, 3, 23)]
# d = d[d['start_time'] >= datetime(2020, 3, 22)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
#
# d = da[da['start_time'] < datetime(2020, 3, 31, 10)]
# d = d[d['start_time'] >= datetime(2020, 3, 30, 10)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
#
# d = da[da['start_time'] < datetime(2020, 3, 31, 18)]
# d = d[d['start_time'] >= datetime(2020, 3, 31, 10)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
#
# d = da[da['start_time'] < datetime(2020, 3, 31, 22)]
# d = d[d['start_time'] >= datetime(2020, 3, 31, 18)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
#
# d = da[da['start_time'] < datetime(2020, 4, 2, 15)]
# d = d[d['start_time'] >= datetime(2020, 3, 31, 22)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
#
# d = da[da['start_time'] < datetime(2020, 4, 8, 10, 46)]
# d = d[d['start_time'] >= datetime(2020, 4, 3, 14, 45)]
# gmv_a = int(d['对照组_成交金额'].sum())
# gmv_b = int(d['实验组_成交金额'].sum())
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击次数'].sum() / d['对照组_浏览量'].sum()
# gmv_b = d['实验组_直播间点击次数'].sum() / d['实验组_浏览量'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_直播间点击人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_直播间点击人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)
# gmv_a = d['对照组_成交人数'].sum() / d['对照组_浏览人数'].sum()
# gmv_b = d['实验组_成交人数'].sum() / d['实验组_浏览人数'].sum()
# gmv_incr = (gmv_b - gmv_a) / gmv_a
# print(gmv_a, gmv_b, gmv_incr)

# print(d.columns)
# print(d.head(10))
import json
from sklearn.metrics.pairwise import pairwise_distances

path = 'C:/Users/99263/IdeaProjects/wwdz/'
file_name = 'i2i-follow_order_cross-expose_punish'
file_name = 'i2i-order-expose_punish'
file_name = 'i2i-follow-expose_punish'
file_name = 'i2i-follow-order'
file_name = 'i2i-10-min-log'

df = pd.read_csv(path + file_name + '.csv')
df = df[['pk', 'json_data']]
df['rec'] = df['json_data'].map(lambda x: json.loads(x).get('recommend'))
df['rec'] = df['rec'].map(lambda x: [list(i.keys())[0] for i in x])
df['pk'] = df['pk'].map(lambda x: x.split(',')[0])
df = df[['pk', 'rec']].values.tolist()
data = {}
for i in df:
    data[i[0]] = i[1]
df = pd.DataFrame(columns=tuple(data.keys()))
for i in data.items():
    df[i[0]] = i[1]
jac_sim = 1 - pairwise_distances(df.T, metric="hamming")
jac_sim = pd.DataFrame(jac_sim, index=df.columns, columns=df.columns)
mean = jac_sim.mean().mean()
print(mean)
print()
