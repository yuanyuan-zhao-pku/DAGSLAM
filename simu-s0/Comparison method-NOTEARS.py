"""
DAG Structure Learning using NOTEARS
References: https://github.com/xunzheng/notears
"""

from notears.linear import notears_linear
from notears import utils
from _dagslam.DAGSLAM import count_accuracy, count_accuracy_und
import os
import csv
import time
import logging
import numpy as np
import pandas as pd

# Set log configuration
logging.basicConfig(
    filename="output-s0-NOTEARS.log",
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

default_save_path = "."

file_acc_s0 = os.path.join(default_save_path, "acc_s0_NOTEARS.csv")

d = 20
with open(file_acc_s0, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "s0",
            "fdr",
            "tpr",
            "fpr",
            "shd",
            "pred_size",
            "F1_score",
            "fdr_und",
            "tpr_und",
            "fpr_und",
            "shd_und",
            "pred_size",
            "F1_score_und",
        ]
    )
    for n in [1000]:
        for s0 in [10, 20, 40]:
            start_time = time.perf_counter()
            ACC = np.zeros([10, 6])
            ACC_und = np.zeros([10, 6])
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_s0{s0}.csv")
            B_true = pd.read_csv(filenameB, header=None)
            B_true = B_true.values
            for i in range(10):
                filenameWE = os.path.join(
                    default_save_path, f"W_est_n{n}_s0{s0}_i{i}.csv"
                )
                filenameX = os.path.join(default_save_path, f"X_n{n}_s0{s0}_i{i}.csv")

                X = pd.read_csv(filenameX, header=None)
                X = X.values
                W_est = notears_linear(X, lambda1=0.1, loss_type="l2")
                assert utils.is_dag(W_est)
                np.savetxt(filenameWE, W_est, delimiter=",")
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
            end_time = time.perf_counter()
            exe_time = (end_time - start_time) / 10
            print(f"s0={s0} execution time: {exe_time}")
            logging.info(f"s0={s0} execution time: {exe_time}")
            acc = np.mean(ACC, axis=0)
            acc_und = np.mean(ACC_und, axis=0)
            print(f"n={n},s0={s0},acc={acc}")
            print(f"n={n},s0={s0},acc_und={acc_und}")
            logging.info(f"n={n},s0={s0},acc={acc}")
            logging.info(f"n={n},s0={s0},acc_und={acc_und}")
            writer.writerow(
                [
                    n,
                    s0,
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    acc[4],
                    acc[5],
                    acc_und[0],
                    acc_und[1],
                    acc_und[2],
                    acc_und[3],
                    acc_und[4],
                    acc_und[5],
                ]
            )
