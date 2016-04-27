#! /usr/bin/env python

import numpy
import gpsdio
import datetime
import sys

cols = ["lat", "lon", "classification","distance_to_port","distance_to_shore","measure_course_diff","measure_courseavg_10800","measure_courseavg_1800","measure_courseavg_21600","measure_courseavg_3600","measure_courseavg_43200","measure_courseavg_86400","measure_coursestddev_10800","measure_coursestddev_1800","measure_coursestddev_21600","measure_coursestddev_3600","measure_coursestddev_43200","measure_coursestddev_86400","measure_distance_to_port","measure_heading_diff","measure_new_score_10800","measure_new_score_1800","measure_new_score_21600","measure_new_score_3600","measure_new_score_43200","measure_new_score_86400","measure_pos_10800","measure_pos_1800","measure_pos_21600","measure_pos_3600","measure_pos_43200","measure_pos_86400","measure_speed","measure_speed_diff","measure_speedavg_10800","measure_speedavg_1800","measure_speedavg_21600","measure_speedavg_3600","measure_speedavg_43200","measure_speedavg_86400","measure_speedstddev_10800","measure_speedstddev_1800","measure_speedstddev_21600","measure_speedstddev_3600","measure_speedstddev_43200","measure_speedstddev_86400","speed","heading","course","measure_course","timestamp","timestamp_diff"]

length = 0

with gpsdio.open(sys.argv[1]) as f:
     for row in f:
          length += 1

x = numpy.zeros(length, dtype=[(name, "f8") for name in cols + ['mmsi']])


segids = {}
segid_counter = 0

with gpsdio.open(sys.argv[1]) as f:
     for rownum, row in enumerate(f):
          for col in cols:
               val = row.get(col, numpy.Infinity)
               if isinstance(val, datetime.datetime):
                    val = float(val.strftime("%s"))
               x[col][rownum] = val

          val = row['mmsi']
          if 'seg_id' in row:
               seg_id = row['seg_id']
               if seg_id not in segids:
                    segids[seg_id] = segid_counter
                    segid_counter += 1
               # Hack to create unique segment id:s and store them as
               # part of the mmsi column. The added 1 in the end is to
               # handle trailing zeros, e.g. distinguish e.g. 20 from 2.
               val += float("0.%s1" % segid_counter)
          x['mmsi'][rownum] = val

numpy.savez_compressed(
    sys.argv[2],
    x=x)
