#! /bin/bash

rm datasets/alex/classified.msg datasets/alex/classified.json datasets/alex/schema.json datasets/alex/classified-filtered.msg datasets/alex/classified-filtered.csv datasets/alex/classified-filtered.npz

python scripts/join_alex_classification.py datasets/alex/tracks.msg datasets/alex/classification-hourlyResultsAll.txt datasets/alex/classified.msg
benthos etl filtersplit -f "row.get('classification', None) is not None" datasets/alex/classified.msg datasets/alex/classified-filtered.msg
python scripts/tonumpy.py datasets/alex/classified-filtered.msg datasets/alex/classified-filtered.npz
