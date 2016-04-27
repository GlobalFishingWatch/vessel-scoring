#! /usr/bin/env python

import numpy
import csv
import datetime
import sys
import os.path

def get_time(r):
    if 'DATETIME' in r:
        return float(datetime.datetime.strptime(r['DATETIME'], "%Y-%m-%d %H:%M:%S").strftime("%s"))        
    elif 'TIME' in r:
        return float(datetime.datetime.strptime(r['TIME'], "%Y%m%d_%H%M%S").strftime("%s"))
    else:
        assert False, "NO TIME: %s" % r

colmap = {
  "mmsi": lambda r: float(r['MMSI']),
  "lon": lambda r: float(r['LONGITUDE']),
  "lat": lambda r: float(r['LATITUDE']),
  "course": lambda r: float(r['COG']),
  "speed": lambda r: float(r['SOG']),
  "timestamp": get_time,
  "classification": lambda r: float(r['COARSE_FIS'])
}

def get_file_size(path):
    length = 0
    with open(path) as f:
         f = csv.DictReader(f)
         for row in f:
              length += 1
    return length

def convert(path, outpath = "classified-filtered.npz"):

    if os.path.isdir(path):
        paths = [os.path.join(path, p) for p in os.listdir(path) if p.endswith('.csv')]
    else:
        paths = [path]

    length = 0
    for path in paths:
        length += get_file_size(path)

    x = numpy.zeros(length, dtype=[(name, "f8") for name in colmap.keys()])

    pos = 0
    for path in paths:
        with open(path) as f:
            f = csv.DictReader(f)
            for row in f:
                for col, mapper in colmap.iteritems():
                    try:
                        val = mapper(row)
                    except:
                        val = numpy.Infinity
                    x[col][pos] = val
                pos += 1

    numpy.savez_compressed(outpath, x=x)

convert(sys.argv[1], sys.argv[2])
