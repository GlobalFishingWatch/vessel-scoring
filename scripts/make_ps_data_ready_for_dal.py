import numpy as np
import pandas as pd
import datetime

ra = np.load("datasets/kristina_ps.measures.npz")['x']
names = ra.dtype.names
columns = {nm : ra[nm] for nm in names}
df = pd.DataFrame(columns)
dates = [datetime.datetime.utcfromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S") for x in df['timestamp']]
df['date'] = dates
df['SOG'] = df.speed
df['LONGITUDE'] = df.lon
df['LATITUDE'] = df.lat
# We aren't using the distshore in either model so set to large value for comparision
df['distshore'] = 10000
#
df.to_csv("datasets/kristina_ps.measures.from_npz.csv")


