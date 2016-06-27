import vessel_scoring.legacy_heuristic_model
import vessel_scoring.random_forest_model
import vessel_scoring.logistic_model
import vessel_scoring.utils
import vessel_scoring.data
import vessel_scoring.add_measures
import vessel_scoring.colspec

import json
import os.path
import numpy
import importlib
import datetime


all_data = ['kristina_trawl', 'kristina_longliner', 'kristina_ps'] + ['slow-transits'] * 10
colspec={"windows": vessel_scoring.colspec.Colspec.windows}

untrained_models = {
    'Logistic':               {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6), 'data': all_data},

    'Logistic--Longliner':    {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
                               'data': ['kristina_longliner'] + ['slow-transits'] * 10},
    'Logistic--Trawler':      {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
                               'data': ['kristina_trawl'] + ['slow-transits'] * 10},

    # 'Logistic--Purse seine OLD': {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
    #                               'data': ['kristina_ps'] + ['slow-transits'] * 10},

    'Logistic--Purse seine':  {
        'model': vessel_scoring.logistic_model.LogisticModel(
            colspec={
                "windows": vessel_scoring.colspec.Colspec.windows,
                "window_measures": vessel_scoring.colspec.Colspec.window_measures + ["measure_daylightavg"],
                "measures": vessel_scoring.colspec.Colspec.measures + ["measure_speed", "measure_daylight"]
                },
            order=3, cross=2),
        'data': ['kristina_ps'] + ['slow-transits'] * 10},

    'Logistic opt MSE':       {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=4, cross=3), 'data': all_data},
    'Random Forest':          {'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=colspec), 'data': all_data},
    'Legacy':                 {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=3600), 'data': all_data},
    "Legacy (12 Hour)":       {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=43200), 'data': all_data},

#     'Logistic (MW)':        {'model': vessel_scoring.logistic_model.LogisticModel(windows=[1800, 3600, 10800, 21600, 43200, 86400], order=6), 'data': all_data},
#     'Logistic (MW/cross3)': {'model': vessel_scoring.logistic_model.LogisticModel(windows=[1800, 3600, 10800, 21600, 43200, 86400], order=6, cross=2), 'data': all_data},
#     'Random Forest (MW)':   {'model': vessel_scoring.random_forest_model.RandomForestModel(windows=[1800, 3600, 10800, 21600, 43200, 86400]), 'data': all_data},
#     "Legacy (3 Hour)":      {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=10800), 'data': all_data},
#     "Legacy (24 Hour)":     {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=86400),   'data': all_data},
}

def load_data():
    datasets = {}
    for filename in os.listdir("datasets"):
        if filename.endswith('.measures.npz'):
            name = filename[:-len('.measures.npz')]
            datasets[name] = dict(zip(['all', 'train', 'cross', 'test'], vessel_scoring.data.load_dataset_by_vessel('datasets/' + filename)))
    return datasets



def train_model_on_data(model, train_data):
    """train `model` with `train_data`

    example:

    model = train_model_on_data(LogisticModel(colsec=dict(window=3600)), train_data)


    """
    y_train = vessel_scoring.utils.is_fishy(train_data)
    model.fit(train_data, y_train)
    return model

def train_model(name, spec, dataset):
    print "Training %s..." % name
    training_data = ([dataset[name]['train']
                      for name in spec['data']]
                     + [dataset[name]['cross']
                        for name in spec['data']])
    training_data = vessel_scoring.utils.concatenate_different_recarrays(training_data)
    return train_model_on_data(spec['model'], training_data)

models_path = os.path.join(os.path.dirname(__file__), "models")

def train_models(models = None, train = None, save=True):
    if models is None:
        models = untrained_models
    if train is None:
        train = load_data()
    trained_models = {name: train_model(name, spec, train)
                      for (name, spec) in models.iteritems()}
    if save:
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        for (name, model) in trained_models.iteritems():
            res = model.dump_dict()
            if res is not None:
                with open(os.path.join(models_path, "%s.json" % name), "w") as f:
                    json.dump(res, f)
    return trained_models


def load_models():
    res = {}

    for filename in os.listdir(models_path):
        with open(os.path.join(models_path, filename)) as f:
            conf = json.load(f)
        name = filename.rsplit(".", 1)[0]

        modulename, clsname = conf['model'].rsplit('.', 1)
        modelcls = getattr(importlib.import_module(modulename), clsname)

        res[name] = modelcls(**conf['args'])
    return res
