class Colspec(object):

    window_measures = ["measure_speedavg", "measure_speedstddev", "measure_coursestddev"]
    windows = [1800, 3600, 10800, 21600, 43200, 86400]
    measures = [] # ["speed"]

    def __init__(self, measures = None, windows = None, window_measures = None):
        if measures is not None: self.measures = measures
        if windows is not None: self.windows = windows
        if window_measures is not None: self.window_measures = window_measures

    def get_cols(self, x):
        colnames = list(self.measures)

        for window in self.windows:
            for measure in self.window_measures:
                colnames.append('%s_%s' % (measure, window))

        cols = [x[col] for col in colnames]

        return cols

    def dump_arg_dict(self):
        return {
            "measures": self.measures,
            "windows": self.windows,
            "window_measures": self.window_measures
            }
