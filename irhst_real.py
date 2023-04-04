import typing

from river import anomaly
import numpy as np
from river import stream
import pandas as pd
from river.evaluate import ADWINMassEstimator, BasicMassEstimator, WindowMassEstimator, EWMAMassEstimator
from river.stats import Mean
from sklearn.metrics import roc_auc_score
import sys
import os
import argparse
from sklearn.metrics import recall_score, precision_score, cohen_kappa_score, f1_score

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, type=str)
parser.add_argument("-n", "--num_trees", required=False, type=int, default=25)
parser.add_argument("-H", "--height", required=False, type=int, default=15)
parser.add_argument("-r", "--runs", required=False, type=int, default=1)
parser.add_argument("-s", "--start", required=False, type=int, default=None)
parser.add_argument("-e", "--end", required=False, type=int, default=None)
parser.add_argument("--estimator", required=False, type=str, default=None)
parser.add_argument("-w", "--window-size", required=False, type=int, default=250)
parser.add_argument("-l", "--leaf-scoring", required=False, type=bool, default=False)
parser.add_argument("-a", "--alpha", required=False, type=float, default=0.01)
parser.add_argument("-i", "--incremental", required=False, type=bool, default=False)
parser.add_argument("-f", "--factor", required=False, type=float, default=2)
parser.add_argument("-p", "--subspace-size", required=False, default=1)
parser.add_argument("-c", "--count-path", required=False, default=False)
args = parser.parse_args()


def accuracy_value(scores, y_true, num):
    tn = 0
    fp = 0
    tp = 0
    fn = 0
    ranks = np.argsort(scores)
    for rank in ranks[-num:]:
        if y_true.iloc[rank] != 0:
            tp += 1
        else:
            fp += 1
    for rank in ranks[:-num]:
        if y_true.iloc[rank] != 0:
            fn += 1
        else:
            tn += 1

    print(tp, fp, tn, fn)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.0
    print(f"auc: {auc}")
    return tp, fp, tn, fn, auc


def _generate_max_min(dimensions):
    max_arr = np.zeros(dimensions)
    min_arr = np.zeros(dimensions)
    for q in range(dimensions):
        s_q = np.random.random_sample()
        max_value = max(s_q, 1 - s_q)
        max_arr[q] = s_q + max_value
        min_arr[q] = s_q - max_value

    return max_arr, min_arr


def scoring(X, y, seed=None):
    if args.estimator == 'ADWIN':
        estimator = ADWINMassEstimator()
    elif args.estimator == 'Basic':
        estimator = BasicMassEstimator()
    elif args.estimator == 'EWMA':
        estimator = EWMAMassEstimator(alpha=args.alpha)
    elif args.estimator == 'Window':
        estimator = WindowMassEstimator(size_window=args.window_size)
    else:
        estimator = None

    try:
        subspace_size = int(args.subspace_size)
    except ValueError:
        try:
            subspace_size = float(args.subspace_size)
        except ValueError:
            subspace_size = str(args.subspace_size)

    model = anomaly.HalfSpaceTrees(window_size=-1 if args.incremental else args.window_size, n_trees=args.num_trees,
                                   height=args.height, limits=None, seed=seed,
                                   mass_estimator=estimator, leaf_scoring=args.leaf_scoring)

    # if args.incremental is False and args.window_size > 0:
    #     dataset = stream.iter_pandas(X.iloc[:args.window_size], y.iloc[:args.window_size])
    #     for i, (x, _) in enumerate(dataset):
    #         print('Learning instance %d' % i)
    #         model.learn_one(x)

    dataset = stream.iter_pandas(X, y)
    scores = []
    pred = []
    mean = Mean()
    mse = Mean()
    for i, (x, _) in enumerate(dataset):
        s = model.score_one(x)
        print('Score is %f for instance %d' % (s, i))
        scores.append(float(s))
        model.learn_one(x)
        pred.append(1 if s > (mean.get() + args.factor * (mse.get() ** 0.5)) else 0)
        mean.update(s)
        mse.update((s - mean.get())**2)

    return scores, pred


def measure_result(y_true, y_pred):
    return recall_score(y_true, y_pred), precision_score(y_true, y_pred), cohen_kappa_score(y_true, y_pred), f1_score(y_true, y_pred)

def batch_result(y_true, scores):
    mean = scores.mean()
    std = scores.std()
    y_pred = scores > mean + args.factor * std
    y_pred = y_pred.astype(int)
    return measure_result(y_true, y_pred)

if __name__ == '__main__':
    df = pd.read_csv(args.dataset, header=None)
    X = df.iloc[:, :-1]  # features
    y = df.iloc[:, -1].astype(int)  # label

    X = X[args.start:args.end]
    y = y[args.start:args.end]

    X = (X - X.min()) / (X.max() - X.min())  # normalize

    print(X.shape)
    print(y.shape)
    anomalies = y.to_numpy().nonzero()[0]
    print(anomalies.shape)

    results = []
    for i in range(args.runs):
        final_scores, y_pred = scoring(X, y, seed=i)
        r = accuracy_value(np.array(final_scores), y, anomalies.shape[0])
        results.append(r)
        print("recall, precision, kappa, f1")
        print(f"incremental: {measure_result(y.to_numpy(), np.array(y_pred))}")
        print(f"batch: {batch_result(y.to_numpy(), np.array(final_scores))}")

    filename, file_extension = os.path.splitext(sys.argv[1])
    results_df = pd.DataFrame(results, columns=['tp', 'fp', 'tn', 'fn', 'auc'])
    # results_df.to_csv(f'{filename}_result.csv', index=False)

    print(args)
