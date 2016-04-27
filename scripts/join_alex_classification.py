import csv
import gpsdio
import datetime
import sys
import math

names=set()
classifications = {}
with open(sys.argv[2]) as f:
    for row in csv.DictReader(f, skipinitialspace=True):
        mmsi = row['mmsi'] = int(row['mmsi'])
        if mmsi not in classifications: classifications[mmsi] = {}
        t = row['start_hour_ms'] = datetime.datetime.utcfromtimestamp(float(row['start_hour_ms'])/1000.0)
        if t not in classifications[mmsi]: classifications[mmsi][t] = set()
        classifications[mmsi][t].add(row['classification'])
        names.add(row['classification'])


with gpsdio.open(sys.argv[3], "w") as outf:
    with gpsdio.open(sys.argv[1]) as f:
        c = 0
        for row in f:
            mmsiclass = classifications[int(row['mmsi'])]
            keys = [key for key in mmsiclass.iterkeys()
                   if key <= row['timestamp'] and key + datetime.timedelta(hours=1) >= row['timestamp']]   
            if keys:
                key = keys[0]
                clss = mmsiclass[key]
                total = len(clss)
                if total > 0:
                    fishing = len([cls for cls in clss if cls and cls != 'Not fishing'])
                    longliner = len([cls for cls in clss if cls == 'Longliner'])
                    purse_seine = len([cls for cls in clss if cls == 'Purse seine'])
                    row['classification'] = float(fishing) / float(total)
                    row['classification_longliner'] = float(longliner) / float(total)
                    row['classification_purse_seine'] = float(purse_seine) / float(total)
            for key in row:
                if isinstance(row[key], float) and (math.isnan(row[key]) or math.isinf(row[key])):
                    row[key] = None
            outf.write(row)
            c += 1
            if c % 1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
