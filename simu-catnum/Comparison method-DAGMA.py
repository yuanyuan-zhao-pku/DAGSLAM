"""
DAG Structure Learning using DAGMA
References: https://github.com/kevinsbello/dagma
"""

# import torch
# from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from dagma import utils
from dagma.linear import DagmaLinear


from _dagslam.DAGSLAM import count_accuracy, count_accuracy_und
import os
import csv
import time
import logging
import numpy as np
import pandas as pd

# Set log configuration
logging.basicConfig(
    filename="output-catnum-DAGMA.log",
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

default_save_path = "."

file_acc_m = os.path.join(default_save_path, "acc_m_DAGMA.csv")

d = 20
with open(file_acc_m, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "m",
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
        for m in [3, 4]:
            start_time = time.perf_counter()
            ACC = np.zeros([10, 6])
            ACC_und = np.zeros([10, 6])
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_m{m}.csv")
            B_true = pd.read_csv(filenameB, header=None)
            B_true = B_true.values
            for i in range(10):
                filenameWE = os.path.join(
                    default_save_path, f"W_est_n{n}_m{m}_i{i}.csv"
                )
                filenameX = os.path.join(default_save_path, f"X_n{n}_m{m}_i{i}.csv")
                X = pd.read_csv(filenameX, header=None)
                X = X.values
                model = DagmaLinear(
                    loss_type="l2"
                )  # create a linear model with least squares loss
                W_est = model.fit(
                    X, lambda1=0.02
                )  # fit the model with L1 reg. (coeff. 0.02)
                np.savetxt(filenameWE, W_est, delimiter=",")
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
            end_time = time.perf_counter()
            exe_time = (end_time - start_time) / 10
            print(f"m={m} execution time: {exe_time}")
            logging.info(f"m={m} execution time: {exe_time}")
            acc = np.mean(ACC, axis=0)
            acc_und = np.mean(ACC_und, axis=0)
            print(f"n={n},m={m},acc={acc}")
            print(f"n={n},m={m},acc_und={acc_und}")
            logging.info(f"n={n},m={m},acc={acc}")
            logging.info(f"n={n},m={m},acc_und={acc_und}")
            writer.writerow(
                [
                    n,
                    m,
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
