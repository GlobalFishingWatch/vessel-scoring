import vessel_scoring.legacy_heuristic_model
import vessel_scoring.random_forest_model
import vessel_scoring.logistic_model
import vessel_scoring.utils
import vessel_scoring.data
import vessel_scoring.evaluate_model

import json
import os.path
import numpy
import importlib

untrained_models = [
    ('Logistic',               vessel_scoring.logistic_model.LogisticModel(windows=[43200], order=6)),
    ('Logistic opt MSE',       vessel_scoring.logistic_model.LogisticModel(windows=[43200], order=4, cross=3)),
#     ('Logistic (MW)',        vessel_scoring.logistic_model.LogisticModel(windows=[1800, 3600, 10800, 21600, 43200, 86400], order=6)),
#     ('Logistic (MW/cross3)', vessel_scoring.logistic_model.LogisticModel(windows=[1800, 3600, 10800, 21600, 43200, 86400], order=6, cross=2)),
    ('Random Forest',          vessel_scoring.random_forest_model.RandomForestModel(windows=[43200])),
#     ('Random Forest (MW)',   vessel_scoring.random_forest_model.RandomForestModel(windows=[1800, 3600, 10800, 21600, 43200, 86400])),
    ('Legacy',                 vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=3600)),
#     ("Legacy (3 Hour)",      vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=10800)),
    ("Legacy (12 Hour)",       vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=43200)),
#     ("Legacy (24 Hour)",     vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=86400)),  
]

def get_default_training_data(transit_weight = 10):
    _, xtrain_trawl, xcross_trawl, xtest_trawl = vessel_scoring.data.load_dataset_by_vessel('datasets/kristina_trawl.measures.npz')
    _, xtrain_lline, xcross_lline, xtest_lline = vessel_scoring.data.load_dataset_by_vessel('datasets/kristina_longliner.measures.npz')
    _, xtrain_pseine, xcross_pseine, xtest_pseine = vessel_scoring.data.load_dataset_by_vessel('datasets/kristina_ps.measures.npz')

    _, xtrain_tran, xcross_tran, xtest_tran = vessel_scoring.data.load_dataset_by_vessel('datasets/slow-transits_scored.measures.npz', even_split=False)
    xtrain_tran = vessel_scoring.utils.clone_subset(xtrain_tran, xtrain_trawl.dtype)
    xcross_tran = vessel_scoring.utils.clone_subset(xcross_tran, xtrain_trawl.dtype)
    xtest_tran = vessel_scoring.utils.clone_subset(xtest_tran, xtrain_trawl.dtype)

    xtrain = numpy.concatenate([xtrain_trawl, xtrain_lline, xtrain_pseine] + [xtrain_tran] * transit_weight)
    xcross = numpy.concatenate([xcross_trawl, xcross_lline, xcross_pseine] + [xcross_tran] * transit_weight)

    return numpy.concatenate([xtrain, xcross])


def train_models(models = None, train = None, save=True):
    if models is None:
        models = untrained_models
    if train is None:
        train = get_default_training_data()
    trained_models = [(name, vessel_scoring.evaluate_model.train_model(mdl, train))
                      for (name, mdl) in models]
    if save:
        if not os.path.exists("models"):
            os.mkdir("models")
        for (name, model) in trained_models:
            if hasattr(model, 'dump_dict'):
                with open("models/%s.json" % name, "w") as f:
                    model_class = type(model)
                    json.dump({'model': "%s.%s" % (model_class.__module__, model_class.__name__),
                               'args': model.dump_dict(),
                               },
                              f)
    return trained_models


def load_models():
    res = {}
    for filename in os.listdir("models"):
        with open(os.path.join("models", filename)) as f:
            conf = json.load(f)
        name = filename.rsplit(".", 1)[0]

        modulename, clsname = conf['model'].rsplit('.', 1)
        modelcls = getattr(importlib.import_module(modulename), clsname)

        res[name] = modelcls(**conf['args'])
    return res
