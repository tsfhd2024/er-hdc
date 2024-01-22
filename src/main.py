import argparse
import copy
import warnings
from typing import Tuple

import numpy as np
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
import torch
from tqdm import tqdm

import onlinehd
from tools.clustering import clustering, process
from tools.correction import correct_checksum, correct_clusters
from tools.utils import error_injection_checksum, inject_error, error_injection_clusters

# loads simple mnist dataset

warnings.filterwarnings("ignore")


def load(
    x_train_filepath: str,
    y_train_filepath: str,
    x_test_filepath: str,
    y_test_filepath: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load and preprocess data from the given file paths.

    Parameters:
    - x_train_filepath (str): File path for the training data features.
    - y_train_filepath (str): File path for the training data labels.
    - x_test_filepath (str): File path for the testing data features.
    - y_test_filepath (str): File path for the testing data labels.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following:
        - x (torch.Tensor): Training data features.
        - x_test (torch.Tensor): Testing data features.
        - y (torch.Tensor): Training data labels.
        - y_test (torch.Tensor): Testing data labels.
    """
    # fetches data
    x = np.load(x_train_filepath)
    y = np.load(y_train_filepath)
    x_test = np.load(x_test_filepath)
    y_test = np.load(y_test_filepath)

    x = x.reshape(len(x), -1)
    x_test = x_test.reshape(len(x_test), -1)

    y = y.flatten()
    y_test = y_test.flatten()

    x = x.astype(np.float)
    y = y.astype(np.int)

    scaler = sklearn.preprocessing.Normalizer().fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)

    # changes data to pytorch's tensors
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    return x, x_test, y, y_test


def main():  # sourcery skip: avoid-builtin-shadow  # sourcery skip: avoid-builtin-shadow
    print("Loading...")
    parser = argparse.ArgumentParser(description="Load error resilience HDC settings.")
    parser.add_argument("--x_train_filepath", type=str, required=True, help="data")
    parser.add_argument("--y_train_filepath", type=str, required=True, help="data")
    parser.add_argument("--x_test_filepath", type=str, required=True, help="data")
    parser.add_argument("--y_test_filepath", type=str, required=True, help="data")
    parser.add_argument("--dim", type=int, required=True, default=1e4, help="data")
    parser.add_argument("--epochs", type=int, required=True, default=200, help="epochs")
    parser.add_argument("--lr", type=float, required=True, default=0.035, help="epochs")
    parser.add_argument(
        "--bootstrap", type=float, required=True, default=1.0, help="bootstrap"
    )
    parser.add_argument(
        "--eps", type=float, required=True, default=1e-5, help="minimal distance DBSCAN"
    )
    parser.add_argument(
        "--nbr_cluster",
        type=int,
        required=True,
        default=100,
        help="minimal distance DBSCAN",
    )
    parser.add_argument(
        "--itr", type=int, required=True, default=5, help="number of iteration"
    )
    parser.add_argument(
        "--init_er",
        type=int,
        required=True,
        default=8,
        help="initial error rate exponent",
    )
    parser.add_argument(
        "--final_er",
        type=int,
        required=True,
        default=0,
        help="final error rate exponent",
    )
    parser.add_argument(
        "--Thresh", type=float, required=True, default=100, help="Threshold Clusters"
    )
    parser.add_argument(
        "--weight_cluster",
        type=float,
        required=True,
        default=1.1,
        help="weight cluster",
    )
    parser.add_argument(
        "--encoding_system",
        type=str,
        required=True,
        default="kernel",
        help="encoding system it should belong to [kernel, record, n-gram]",
    )

    args = parser.parse_args()

    x, x_test, y, y_test = load(
        args.x_train_filepath,
        args.y_train_filepath,
        args.x_test_filepath,
        args.y_test_filepath,
    )
    classes = y.unique().size(0)
    features = x.size(1)
    model = onlinehd.OnlineHD(
        classes, features, dim=args.dim, encoding_system=args.encoding_system
    )

    print("Training...")
    org_model = model.fit(
        x, y, bootstrap=args.bootstrap, lr=args.lr, epochs=args.epochs
    )

    N = 10000
    eps = args.eps
    while N > args.nbr_cluster:
        D, N = clustering(org_model.model.numpy(), eps)
        eps *= args.weight_cluster

    checksum = process(org_model.model.numpy())
    min = checksum.min()
    max = checksum.max()
    acc_tests = []
    acc_test_ers = []
    acc_test_news = []
    acc_test_sums = []
    for p in tqdm(range(args.init_er, args.final_er, -1)):
        acc_test = 0
        acc_test_er = 0
        acc_test_new = 0
        acc_test_sum = 0
        for _ in tqdm(range(args.itr)):
            model = copy.deepcopy(org_model)
            D1 = copy.deepcopy(D)
            m = model.model

            m0, _ = inject_error(m, 10 ** (-p))
            # -------------------ScaleHD-Method---------------------------
            maxi = m.max()
            mini = m.min()
            m = torch.clamp(m0, torch.tensor(mini), torch.tensor(maxi))
            model.model = m
            yhat_test = model(x_test)
            acc_test_er += (y_test == yhat_test).float().mean()
            # ------------------Faulty-model-----------------------------
            m = copy.deepcopy(m0)
            model.model = m
            yhat_test = model(x_test)
            acc_test += (y_test == yhat_test).float().mean()

            # ------------------Clustering--------------------------------
            m = copy.deepcopy(m0)
            D1 = error_injection_clusters(D1, 10**(-p))
            m1 = correct_clusters(m.numpy(), D1, args.Thresh)
            model.model = torch.tensor(m1)
            yhat_test = model(x_test)
            acc_test_new += (y_test == yhat_test).float().mean()
            # ------------------Checksum----------------------------------
            m = copy.deepcopy(m0)
            checksum = error_injection_checksum(checksum, 10 ** (-p), min, max)
            m1 = correct_checksum(m.numpy(), checksum)
            model.model = torch.tensor(m1)
            yhat_test = model(x_test)
            acc_test_sum += (y_test == yhat_test).float().mean()

        acc_tests.append(acc_test / args.itr)
        acc_test_ers.append(acc_test_er / args.itr)
        acc_test_news.append(acc_test_new / args.itr)
        acc_test_sums.append(acc_test_sum / args.itr)
        print(f"acc test faulty {acc_test/args.itr}")
        print(f"acc test new {acc_test_new/args.itr}")
        print(f"acc test ers {acc_test_er/args.itr}")
        print(f"acc test sums {acc_test_sum/args.itr}")


if __name__ == "__main__":
    main()
