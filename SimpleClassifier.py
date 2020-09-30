import numpy as np
import matplotlib.pyplot as plt


def alpha(fp, tn):
    return fp / (fp + tn) if fp + tn != 0 else "Calculation impossible"


def beta(fn, tp):
    return fn / (tp + fn) if tp + fn != 0 else "Calculation impossible"


def accuracy(tp, tn, fn, fp):
    return (tp + tn) / (tp + fn + tn + fp) if tp + fn + tn + fp != 0 else "Calculation impossible"


def precision(tp, fp):
    return tp / (tp + fp) if tp + fp != 0 else "Calculation impossible"


def recall(tp, fn):
    return tp / (tp + fn) if tp + fn != 0 else "Calculation impossible"


def f1_score(recall, precision):
    return 2*(recall * precision)/(recall + precision) if recall + precision != 0 else "Calculation impossible"


def print_it_all():
    tp, tn, fn, fp = get_metrics(best_t)

    print("TP ", tp)
    print("TN ", tn)
    print("FN ", fn)
    print("FP ", fp)
    print("AUC ", AUC)
    print("Best t ", best_t)
    print("Alpha ", alpha(fp, tn))
    print("Beta ", beta(fn, tp))
    print("Accuracy ", accuracy(tp, tn, fn, fp))
    print("Precision ", precision(tp, fp))
    print("Recall ", recall(tp, fn))
    print("F1 score ", f1_score(recall(tp, fn), precision(tp, fp)))


def get_metrics(t):
    football_pred = footballHeight <= t
    basketball_pred = basketballHeight > t

    tp = np.sum(football_pred)
    fn = np.sum(1 - football_pred)
    tn = np.sum(basketball_pred)
    fp = np.sum(1 - basketball_pred)
    return tp, fn, tn, fp


def calc_params(t):
    tp_cur, fn_cur, tn_cur, fp_cur = get_metrics(t)

    alphaCur = alpha(fp_cur, tn_cur) if (fp_cur + tn_cur) != 0 else 0
    recallCur = recall(tp_cur, fn_cur) if (tp_cur + fn_cur) != 0 else 0
    accuracyCur = accuracy(tp_cur, tn_cur, fn_cur, fp_cur) if (tp_cur + fn_cur + tn_cur + fp_cur) != 0 else 0

    return alphaCur, recallCur, accuracyCur


def graph():
    plt.figure()
    plt.plot(alphaArr, recallArr, color='red')
    plt.plot([alphaArr[0], alphaArr[iterations.size-1]], [recallArr[0], recallArr[iterations.size-1]], 'b')
    plt.show()


mu_0 = 180
mu_1 = 193
sigma_0 = 6
sigma_1 = 7

footballHeight = np.random.randn(1000) * sigma_0 + mu_0
basketballHeight = np.random.randn(1000) * sigma_1 + mu_1
iterations = np.arange(250, dtype=np.int)

vectorParams = np.vectorize(calc_params)
alphaArr, recallArr, accuracyArr = vectorParams(iterations)

best_t = accuracyArr.argmax(axis=0)

alphaDiff = np.abs(alphaArr[1:] - alphaArr[:iterations.size - 1])
recallSum = recallArr[1:] + recallArr[:iterations.size - 1]

AUC = np.sum((alphaDiff * recallSum / 2))

graph()

print_it_all()
