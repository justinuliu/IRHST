import argparse
import numpy as np
import pandas as pd
from river import anomaly
from river import stream
from sklearn.metrics import roc_auc_score
from river.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_trees", required=False, type=int, default=25)
parser.add_argument("-H", "--height", required=False, type=int, default=15)
parser.add_argument("-t", "--model_type", required=True, type=str, choices=['o', 'l', 'p'])
parser.add_argument("-s", "--seed_range", required=False, type=int, default=10)
parser.add_argument("-i", "--initial_samples", required=False, type=int, default=64)
parser.add_argument("-u", "--update_samples", required=False, type=int, default=32)
parser.add_argument("--update_times", required=False, type=int, default=20)
parser.add_argument("-m", "--anomaly_factor", required=True, type=float)
parser.add_argument("-o", "--output_name", required=True, type=str)
parser.add_argument("-d", "--dimension", required=True, type=int)

args = parser.parse_args()


def KLD(u0, s0, u1, s1):
    # general Kullback-Leibler divergence for the multivariate normal distribution
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    return ((np.linalg.inv(s1) @ s0).trace() - len(u0) + (u1 - u0) @ np.linalg.inv(s1) @ (u1 - u0).T + np.log(
        np.linalg.det(s1) / np.linalg.det(s0))) / 2


def sKL(u0, s0, u1, s1):
    return KLD(u0, s0, u1, s1) + KLD(u1, s1, u0, s0)


def AUC(model_type, training_size, update_list, mean0, cov0, mean1, cov1, seeds, output_name, n_trees=25, height=15, target_fpr=0.01):
    test_size = 5000
    val_size = 1000

    results = []
    skl = sKL(mean0, cov0, mean1, cov1)
    print(f"sKL: {skl}")

    for seed in seeds:
        np.random.seed(seed)
        test_X0 = np.random.multivariate_normal(mean0, cov0, test_size)
        test_X1 = np.random.multivariate_normal(mean1, cov1, test_size)
        val_X0 = np.random.multivariate_normal(mean0, cov0, val_size)
        test_X = np.concatenate((test_X0, test_X1))
        test_Y = np.concatenate((np.zeros(test_size), np.ones(test_size)))

        train_X = np.random.multivariate_normal(mean0, cov0, training_size)
        dataset = stream.iter_array(train_X, np.zeros(training_size))

        min_max_scaler = MinMaxScaler()
        for x, _ in dataset:
            min_max_scaler.learn_one(x)

        limits = {key: (min_max_scaler.min[key].get(), min_max_scaler.max[key].get()) for key in
                  min_max_scaler.min.keys()}

        if model_type == 'o':
            model = anomaly.HalfSpaceTrees(window_size=training_size, n_trees=n_trees, height=height, limits=limits,
                                           seed=seed)
        elif model_type == 'p':
            model = anomaly.HalfSpaceTrees(window_size=-1, n_trees=n_trees, height=height, limits=limits, seed=seed,
                                           leaf_scoring=False)
        elif model_type == 'l':
            model = anomaly.HalfSpaceTrees(window_size=-1, n_trees=n_trees, height=height, limits=limits, seed=seed,
                                           leaf_scoring=True)
        else:
            raise Exception(f"Unknown model type: {model_type}")

        dataset = stream.iter_array(train_X, np.zeros(training_size))
        for x, _ in dataset:
            model.learn_one(x)

        val_scores = []
        dataset = stream.iter_array(val_X0, np.zeros(val_size))
        for x, _ in dataset:
            s = model.score_one(x)
            val_scores.append(s)
        th = val_scores[int(val_size * target_fpr)]

        samples = training_size
        for i, update_size in enumerate(update_list):
            samples += update_size
            update_X = np.random.multivariate_normal(mean0, cov0, update_size)
            dataset = stream.iter_array(update_X, np.zeros(update_size))
            for x, _ in dataset:
                model.learn_one(x)

            scores = []
            dataset = stream.iter_array(test_X, test_Y)
            for x, _ in dataset:
                s = model.score_one(x)
                scores.append(s)

            auc_score = roc_auc_score(test_Y, scores)

            results.append((seed, args.dimension, model_type, samples, args.anomaly_factor, skl, auc_score))

    df = pd.DataFrame(results, columns=['seed', 'dimension', 'model_type', 'num_samples', 'm', 'sKL', 'auc'])
    df.to_csv(output_name, index=False)


if __name__ == '__main__':
    AUC(args.model_type, args.initial_samples, [args.update_samples] * args.update_times, np.ones(args.dimension) * 0.5,
        np.identity(args.dimension) * 0.1, np.ones(args.dimension) * args.anomaly_factor,
        np.identity(args.dimension) * 0.1, seeds=range(10), output_name=args.output_name, n_trees=args.num_trees,
        height=args.height)

    print(args)
