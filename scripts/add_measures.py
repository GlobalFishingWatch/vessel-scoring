# Find the root directory of vessel-scoring, if running directly from git
import sys, os.path; sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import vessel_scoring.add_measures
import numpy


numpy.savez_compressed(
    sys.argv[2],
    x=vessel_scoring.add_measures.add_measures(
        numpy.load(sys.argv[1])['x'],
        verbose=True,
        err=sys.stderr))
