"""Calculate accuracy metrics for each replication."""

import logging
import os
import time
import numpy as np
import pandas as pd
import csv
from _dagslam.DAGSLAM import count_accuracy, count_accuracy_und

default_save_path = "."

file_acc_graphmodel_each = os.path.join(
    default_save_path, "each_acc_graphmodel_DAGSLAM.csv"
)
# file_acc_graphmodel_each = os.path.join(
#     default_save_path, "each_acc_graphmodel_NOTEARS.csv"
# )
# file_acc_graphmodel_each = os.path.join(
#     default_save_path, "each_acc_graphmodel_DAGMA.csv"
# )

with open(file_acc_graphmodel_each, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "graphmodel",
            "i",
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
        for graph_type in ["ER", "SF"]:
            start_time = time.perf_counter()
            ACC = np.zeros([10, 6])
            ACC_und = np.zeros([10, 6])
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_{graph_type}.csv")
            B_true = pd.read_csv(filenameB, header=None)
            B_true = B_true.values
            for i in range(10):
                filenameWE = os.path.join(
                    default_save_path, f"W_est_n{n}_{graph_type}_i{i}.csv"
                )
                filenameX = os.path.join(
                    default_save_path, f"X_n{n}_{graph_type}_i{i}.csv"
                )
                X = pd.read_csv(filenameX, header=None)
                X = X.values
                W_est = pd.read_csv(filenameWE, header=None)
                W_est = W_est.values
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
                writer.writerow(
                    [
                        n,
                        graph_type,
                        i,
                        ACC[i, 0],
                        ACC[i, 1],
                        ACC[i, 2],
                        ACC[i, 3],
                        ACC[i, 4],
                        ACC[i, 5],
                        ACC_und[i, 0],
                        ACC_und[i, 1],
                        ACC_und[i, 2],
                        ACC_und[i, 3],
                        ACC_und[i, 4],
                        ACC_und[i, 5],
                    ]
                )
            end_time = time.perf_counter()
            exe_time = end_time - start_time
            print(f"graph_type={graph_type} execution time: {exe_time}")
            logging.info(f"graph_type={graph_type} execution time: {exe_time}")
