[![Build Status](https://travis-ci.org/GlobalFishingWatch/vessel-scoring.svg?branch=master)](https://travis-ci.org/GlobalFishingWatch/vessel-scoring)

[![Coverage Status](https://coveralls.io/repos/github/GlobalFishingWatch/vessel-scoring/badge.svg?branch=master)](https://coveralls.io/github/GlobalFishingWatch/vessel-scoring?branch=master)

# About
This repository contains fishing scoring heuristics and evaluation of their effectiveness,
as well as development of supervised-ML scoring algorithms.

## Training data
Training data for this project is too big to store in this repository. It will be licensed separately here:
https://github.com/GlobalFishingWatch/training-data

You do not need the training data to use the prediction models as the trained model parameters are stored in the repository.

## Tools
It also contains some tools to deal with the data formats of some of our partners:

* Kristinas CSV-based format
* Alex' and Chris' fishing-hour sidecar format

## Docker

To run a Jupyter notebook:

`$DATADIR` is the path to the directory storing the necessary data files

```
DATADIR=$HOME/datasets

tree $DATADIR
datasets
├── classified-filtered.measures.npz
├── false_positives.measures.npz
├── kristina_longliner.measures.npz
├── kristina_ps.measures.npz
└── kristina_trawl.measures.npz

docker pull etheleon/pirates:latest
docker run --rm -it
    -v $HOME:/w \
    -v $DATADIR:/w/datasets
    -w /w \
    -p 8888:8888 etheleon/pirates:latest \
    jupyter notebook --no-browser --ip="*" --allow-root
```

Then go to `http://<DOCKER-MACHINE-IP>:8888`

## API

    import vessel_scoring.models
    models = vessel_scoring.models.load_models()
    is_fishing = models['Logistic'].predict_proba(track_points)[:,1]

## Copyright 2016 SkyTruth
Authors: Egil Möller <egil@skytruth.org>, Timothy Hochberg <tim.hochberg@IEEE.ORG>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
