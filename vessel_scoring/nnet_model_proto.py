from lasagne.layers import (InputLayer, DenseLayer, Conv1DLayer,
                            DropoutLayer,
                            ConcatLayer, MaxPool1DLayer, ReshapeLayer,
                            LSTMLayer, SliceLayer, BatchNormLayer,
                            NonlinearityLayer,NINLayer, FeaturePoolLayer,
                            GlobalPoolLayer, ParametricRectifierLayer,
                            GaussianNoiseLayer)
from lasagne.nonlinearities import leaky_rectify, softmax, sigmoid
from lasagne import utils, init
from theano import tensor as T
import theano
from nolearn.lasagne import NeuralNet, BatchIterator, TrainSplit
from lasagne.objectives import squared_error, categorical_crossentropy, binary_crossentropy
from lasagne.updates import nesterov_momentum
import numpy as np
import numpy as np
# from vessel_scoring.utils import get_polynomial_cols

def get_polynomial_cols(x, windows):
    colnames = []
    #colnames.append("speed")
    for window in windows:
        colnames.append('measure_speedavg_%s' % window)
        colnames.append('measure_speedstddev_%s' % window)
        colnames.append('measure_coursestddev_%s_log' % window)
    cols = [x[col] for col in colnames]
    return cols


import vessel_scoring.base_model

N_WINDOWS = 6
WIDTH = 2048
N_BASE_FEATURES = 3
N_FEATURES = N_WINDOWS * N_BASE_FEATURES

def L(layer, **kwargs):
    """Create (Layer {kwargs}) pairs for nolearn"""
    return (layer, kwargs)

class OnEpochFinished:

    def __init__(self, learning_rate, decay_scale):
        self.learning_rate = learning_rate
        self.decay_scale = decay_scale

    def __call__(self, nn, train_history):
        new_lr = self.learning_rate.get_value() * self.decay_scale
        self.learning_rate.set_value(np.float32(new_lr))


def make_net(batch_size=128):
    learning_rate = theano.shared(np.float32(0.005))
    momentum = 0.9

    layers = [L(InputLayer, name="input", shape=(None, N_FEATURES)),
              L(DenseLayer, num_units=WIDTH, nonlinearity=leaky_rectify),
              L(DropoutLayer, p=0.5),
              L(DenseLayer, num_units=WIDTH, nonlinearity=leaky_rectify),
              L(DropoutLayer, p=0.5),
              L(DenseLayer, num_units=1, nonlinearity=sigmoid)
              ]

    nnet = NeuralNet(
        y_tensor_type=T.matrix,
        layers=layers,
        regression=True,
        max_epochs=100,
        verbose=1,
        train_split = TrainSplit(eval_size=0.05),
        objective_l2=1e-5,
        objective_loss_function=binary_crossentropy,
        update=nesterov_momentum,
        update_learning_rate=learning_rate,
        update_momentum=momentum,
        on_epoch_finished=[OnEpochFinished(learning_rate, 0.95)],
        custom_scores=[("val acc", lambda t, y:
             np.mean(t == (y > 0.5)))]
        )

    return nnet




class NNetModelProto:

    windows = ['10800', '1800', '21600', '3600', '43200', '86400']

    def __init__(self, batch_size=64, **args):
        """
        windows - list of window sizes to use in features
        See RandomForestClassifier docs for other parameters.
        """
        self.net = make_net(batch_size, )
        if args:
            for key in args:
                args[key] = [np.array(item) for item in args[key]]
            self.net.load_params_from(args)

    def dump_arg_dict(self):
        def convert(data):
            if isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, (list, tuple)):
                return [convert(item) for item in data]
            elif isinstance(data, dict):
                return {key: convert(value)
                        for key, value in data.items()}
            else:
                return data
        return convert(dict(self.get_all_param_values()))

    def _make_features(self, X):
        x = np.transpose(get_polynomial_cols(X, self.windows))
        return (x.astype('float32') - self.mean) / self.std

    def predict_proba(self, X):
        X = self._make_features(X)
        y = np.zeros([len(X), 2], dtype='float32')
        y[:,1] = self.net.predict_proba(X).reshape(-1)
        y[:,0] = 1 - y[:,1]
        return y

    def fit(self, X, y):
        self.mean = 0
        self.std = 1
        X = self._make_features(X)
        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.mean(axis=0, keepdims=True)
        X = (X - self.mean) / self.std
        inds = np.arange(len(X))
        np.random.shuffle(inds)
        return self.net.fit(X[inds], y[inds].astype('float32'))







