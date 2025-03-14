import numpy as np

def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (union_size(yhatmic, ymic, 0) + 1e-10)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-10)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (ymic.sum(axis=0) + 1e-10)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


def all_metrics(y_hat, y):
    """
    :param y_hat:
    :param y:

    :return:
    """
    names = ['acc', 'prec', 'rec', 'f1']
    macro_metrics = all_macro(y_hat, y)

    y_mic = y.ravel()
    y_hat_mic = y_hat.ravel()
    micro_metrics = all_micro(y_hat_mic, y_mic)

    metrics = {names[i] + "_macro": macro_metrics[i] for i in range(len(macro_metrics))}
    metrics.update({names[i] + '_micro': micro_metrics[i] for i in range(len(micro_metrics))})

    return metrics



# 使用pytorch计算top5准确率的函数[^2^][2]
def topk_accuracy(logits, target, topk=(1,5,10)):
    indices = np.argsort(logits, axis=-1)
    batch_size,class_num = logits.shape
    ans = []
    for k in topk:
        predict = np.zeros((batch_size,class_num))
        for i in range(batch_size):
            predict[i,indices[i,-k:]] = 1
        ans.append(np.sum(predict*target) / (batch_size*k))
    return ans

def print_metrics(metrics_test):
    print("\n[MACRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_macro"], metrics_test["prec_macro"], metrics_test["rec_macro"], metrics_test["f1_macro"]))

    print("[MICRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_micro"], metrics_test["prec_micro"], metrics_test["rec_micro"], metrics_test["f1_micro"]))


def write_result(report, result_path):
    with open(result_path, "w", encoding="UTF-8")as f:
        f.write(report)