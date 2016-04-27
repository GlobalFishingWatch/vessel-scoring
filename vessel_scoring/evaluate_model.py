import matplotlib.pyplot as plt
from vessel_scoring import utils
import numpy as np
from sklearn import metrics
from IPython.core.display import display, HTML



def train_model(model, train_data):
    """train `model` with `train_data`

    example:

    model = train_model(LogisticModel(window=3600), train_data)


    """
    y_train = utils.is_fishy(train_data)
    model.fit(train_data, y_train)
    return model


# TODO break this into a plotting and a text part
def evaluate_model(model, test_data, name=None):
    """Plot some graphs and compute some metrics on a model

    model - a trained model
    test_data - data to use on the evalutions

    """
    is_fishy = utils.is_fishy(test_data)

    score = model.predict_proba(test_data)[:,1]
    score_fishy = score[is_fishy]
    score_nonfishy =  score[~is_fishy]

    precisions, recalls, thresholds = metrics.precision_recall_curve(is_fishy, score)


    model_description = str(model) if (name is None) else name
    display(HTML("<h1>%s</h1>" % model_description))

    ylim = 15.0

    f, (a1, a2) = plt.subplots(1, 2, figsize=(20,5))

    a1_precall = a1.twinx()
    def convert_range(ax_f):
        y1, y2 = ax_f.get_ylim()
        a1_precall.set_ylim(y1 / ylim, y2 / ylim)
        a1_precall.figure.canvas.draw()
    a1.callbacks.connect("ylim_changed", convert_range)

    new_score_fishy = a1.hist(score_fishy, bins=200,
            normed=True, color='b', alpha=0.5, label="fishy score")
    new_score_nonfishy = a1.hist(score_nonfishy, bins=200,
            normed=True, color='r', alpha=0.5, label="nonfishy score")

    plot_precision = a1_precall.plot(thresholds, precisions[:-1], color='g', label='Precision')
    plot_recall = a1_precall.plot(thresholds, recalls[:-1], color='b', label='Recall')

    a1.set_ylim(0, ylim)
    a1.set_xlim(0, 1)

    a1.set_ylabel('Histogram count')
    a1.set_xlabel('Prediction score')
    a1_precall.set_ylabel('Curve')

    fpr, tpr, _ = metrics.roc_curve(is_fishy, score)
    auc = metrics.auc(fpr, tpr)

    predicted = score > 0.5
    fp = (predicted & ~(is_fishy)).sum() / float(len(is_fishy))

    lloss = metrics.log_loss(is_fishy, predicted)

    label = 'ROC curve\narea = %0.2f\nlog loss = %0.2f\nfp = %0.2f' % (auc, lloss, fp)
    a2.plot(fpr, tpr, color='r', label=label)
    a2.set_xlabel('False positive rate')
    a2.set_ylabel('True positive rate')

    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a1.get_legend_handles_labels()
    h3, l3 = a1_precall.get_legend_handles_labels()
    a2.legend(h1+h2+h3, l1+l2+l3, loc='lower right')

    plt.show()

    total = sum(new_score_fishy[0] + new_score_nonfishy[0])
    non_overlap = sum(abs(new_score_fishy[0] - new_score_nonfishy[0]))
    overlap = total - non_overlap
    error = overlap / total

def compare_models(models, test_data):
    is_fishy = utils.is_fishy(test_data)

    f, (a1, a2) = plt.subplots(2, 1, figsize=(20,20))

    for (name, mdl) in models:
        score = mdl.predict_proba(test_data)[:,1]
        fpr, tpr, _ = (metrics.roc_curve(is_fishy, score))
        auc = metrics.auc(fpr, tpr)
        a1.plot(fpr, tpr, label='{0}: {1:.3f} AUC'.format(name, auc))
    a1.set_xlabel('False Positive Rate')
    a1.set_ylabel('True Positive Rate')
    a1.legend(loc="lower right")

    for (name, mdl) in models:
        score = mdl.predict_proba(test_data)[:,1]
        precisions, recalls, thresholds = metrics.precision_recall_curve(is_fishy, score)
        a2.plot(recalls, precisions, label='{0}'.format(name))
    a2.set_xlabel('Recall')
    a2.set_ylabel('Precision')
    a2.legend(loc="lower right")

    plt.show()
