"""Simulation for different graph models"""

from _dagslam.DAGSLAM import dagslam, is_dag, count_accuracy, count_accuracy_und

import os
import csv
import time
import logging
import numpy as np
import pandas as pd

default_save_path = "."

file_acc_graphmodel = os.path.join(default_save_path, "acc_graphmodel_DAGSLAM.csv")


def get_loss_type(graphmodel):
    """Define node types: "gauss" for continuous nodes, "logistic" for binary nodes."""
    return ["gauss"] * 18 + ["logistic"] + ["gauss"]


d = 20
with open(file_acc_graphmodel, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "n",
            "graphmodel",
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
                loss_type = get_loss_type(graphmodel=graph_type)
                X = pd.read_csv(filenameX, header=None)
                X = X.values
                W_est = dagslam(
                    X,
                    loss_type=loss_type,
                    m_vec=[1] * d,
                    lambda1=0.1,
                )
                assert is_dag(W_est)
                np.savetxt(filenameWE, W_est, delimiter=",")
                ACC[i, :] = count_accuracy(B_true, W_est != 0)
                ACC_und[i, :] = count_accuracy_und(B_true, W_est != 0)
                print(ACC[i, :])
                print(ACC_und[i, :])
                logging.info(ACC[i, :])
                logging.info(ACC_und[i, :])
            end_time = time.perf_counter()
            exe_time = (end_time - start_time) / 10
            print(f"graph_type={graph_type} execution time: {exe_time}")
            logging.info(f"graph_type={graph_type} execution time: {exe_time}")
            acc = np.mean(ACC, axis=0)
            acc_und = np.mean(ACC_und, axis=0)
            print(f"n={n},graph_type={graph_type},acc={acc}")
            print(f"n={n},graph_type={graph_type},acc_und={acc_und}")
            logging.info(f"n={n},graph_type={graph_type},acc={acc}")
            logging.info(f"n={n},graph_type={graph_type},acc_und={acc_und}")
            writer.writerow(
                [
                    n,
                    graph_type,
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
