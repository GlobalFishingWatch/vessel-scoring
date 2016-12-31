from vessel_scoring import utils
from sklearn import metrics
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML, Markdown
import pandas
import os.path

def evaluate_model(model, test_data, name=None):
    evaluate_score(
        model.predict_proba(test_data)[:,1],
        test_data,
        str(model) if (name is None) else name)


def evaluate_score(score, test_data, name):
    """Plot some graphs and compute some metrics on a set of predictions.  Requires
    matplotlib and IPython.

    score - model predictions, same length as test_data
    test_data - data to use on the evalutions

    """

    is_fishy = utils.is_fishy(test_data)

    score_fishy = score[is_fishy]
    score_nonfishy =  score[~is_fishy]

    precisions, recalls, thresholds = metrics.precision_recall_curve(is_fishy, score)

    display(HTML("<h1>%s</h1>" % name))

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

def compare_models_at_cutoff(models, test_data, predictions = None):
    if not predictions: predictions = {}

    if isinstance(models, dict):
        models = models.items()
        models.sort(lambda a, b: cmp(a[0], b[0]))

    for name, mdl in models:
        predictions[name] = ((mdl.predict_proba(test_data)[:,1] > 0.5), test_data['classification'])

    lines = ["|Model|Recall|Precision|F1-Score|",
         "|-----|------|---------|--------|"]
    for name in sorted(predictions):
        pred, actual = predictions[name]
        lines.append("|{}|{:.2f}|{:.2f}|{:.2f}|".format(
                name, 
                metrics.recall_score(actual, pred),
                metrics.precision_score(actual, pred), 
                metrics.f1_score(actual, pred)))
    display(Markdown('\n'.join(lines)))

def compare_metrics(models, test_data, threshold=0.5):
    labels = utils.is_fishy(test_data)
    for (name, mdl) in models:
        predicted = (mdl.predict_proba(test_data)[:,1] > threshold)
        print(name)
        print(metrics.classification_report(labels, predicted))


def compare_metrics_table(models, test_data, threshold=0.5, cell_width=20):
    labels = utils.is_fishy(test_data)
    items = []
    items.append(['', 'Precision', 'Recall', 'F1-score'])
    items.append(['-' * cell_width] * 4)
    for (name, mdl) in models:
        predicted = (mdl.predict_proba(test_data)[:,1] > threshold)
        precision = metrics.precision_score(labels, predicted)
        recall = metrics.recall_score(labels, predicted)
        f1 =  metrics.f1_score(labels, predicted)
        row = [name]
        row += ["{:.3f}".format(x) for x in [precision, recall, f1]]
        items.append(row)
    text = ''
    for row in items:
        text += ('|' + '|'.join('{0:^{1}}'.format(x, cell_width) for x in row) + '|\n')
    return text


def compare_auc(models, test_data):
    import matplotlib.pyplot as plt

    is_fishy = utils.is_fishy(test_data)

    f, a1 = plt.subplots(1, 1, figsize=(8,4))

    if isinstance(models, dict):
        models = models.items()
        models.sort(lambda a, b: cmp(a[0], b[0]))

    for (name, mdl) in models:
        score = mdl.predict_proba(test_data)[:,1]
        fpr, tpr, _ = (metrics.roc_curve(is_fishy, score))
        auc = metrics.auc(fpr, tpr)
        a1.plot(fpr, tpr, label='{0}: {1:.3f} AUC'.format(name, auc))
    a1.set_xlabel('False Positive Rate')
    a1.set_ylabel('True Positive Rate')
    a1.set_xlim(0,1)
    a1.set_ylim(0,1)
    a1.legend(loc="lower right")
    a1.set_ylim(0, 1)

    plt.show()

def compare_pr(models, test_data):
    import matplotlib.pyplot as plt

    is_fishy = utils.is_fishy(test_data)

    f, (a2) = plt.subplots(1, 1, figsize=(8,4))

    for (name, mdl) in models:
        score = mdl.predict_proba(test_data)[:,1]
        precisions, recalls, thresholds = metrics.precision_recall_curve(is_fishy, score)
        a2.plot(recalls, precisions, label='{0}'.format(name))
    a2.set_xlabel('Recall')
    a2.set_ylabel('Precision')
    a2.set_xlim(0,1)
    a2.set_ylim(0,1)
    a2.legend(loc="lower right")
    a2.set_ylim(0, 1)

    plt.show()


def load_dal_predictions(path):
    if not os.path.exists(path):
        raise IOError(path)
    dal_res = pandas.read_csv(path)
    mask = np.array([(x in test_mmsi) for x in dal_res['mmsi']]).astype(bool)
    res = (dal_res.preds[mask], dal_res.classification[mask])
    # Fix predictions for Trawler.py
    return ([{'F': 1, 'N': 0}.get(x, x) for x in res[0]], res[1])
