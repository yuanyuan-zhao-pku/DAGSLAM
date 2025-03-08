"""  
This algorithm is an extension of the NOTEARS algorithm.  
The original NOTEARS algorithm was developed by Xun Zheng, et al.  
This implementation is authored by Yuanyuan Zhao.  
"""

"""Main function to execute the DAGSLAM algorithm for causal structure learning."""
import numpy as np
import numpy.linalg as la
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
import igraph as ig
import pandas as pd
import csv
from sklearn.preprocessing import OneHotEncoder
import logging

# Set log configuration
logging.basicConfig(
    filename="output-samplesize-dagslam.log",
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def dagslam(
    X,
    loss_type,
    m_vec,
    lambda1=0.03,
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e16,
    w_threshold=0.25,
):
    """Implement the DAGSLAM algorithm for learning the structure of a Bayesian network."""
    n, d = X.shape
    Id = np.eye(d).astype(np.float64)
    w_est = np.zeros(2 * d * d)  # double w_est into (w_pos, w_neg)
    beta_est = np.zeros(
        2 * d * sum(m_vec)
    )  # The coefficient is applicable when multi-nominal nodes are used as child nodes, while the beta values for other node types are zero.
    rho = float(1.0)
    alpha = float(0.0)
    h = np.inf

    def Beta_tolist(x, vec=m_vec):
        """Convert the beta parameters from a matrix format to a list format."""
        split_arrays = np.split(x, np.cumsum(vec), axis=1)[:-1]
        res_list = [arr.squeeze() for arr in split_arrays]
        return res_list

    def adj_beta(beta_est):
        """Convert the doubled beta parameters back to the matrix format."""
        Betas = (beta_est[: d * sum(m_vec)] - beta_est[d * sum(m_vec) :]).reshape(
            [d, sum(m_vec)]
        )
        return Betas

    def adj_w(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[: d * d] - w[d * d :]).reshape([d, d])

    def w_beta_toW(w_est, beta_est, vec=m_vec):
        """Convert the w and beta parameters to the adjacency matrix W."""
        W = (w_est[: d * d] - w_est[d * d :]).reshape([d, d])
        Betas = adj_beta(beta_est)
        Beta_list = Beta_tolist(Betas)
        for j in range(d):
            if m_vec[j] != 1:
                W[:, j] = np.sum(Beta_list[j] ** 2, axis=1)
        return W

    def softmax(x):
        """Compute the softmax of the input array."""
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _loss(w_est, beta_est):
        """Calculate the loss function based on the current parameters."""
        Betas = adj_beta(beta_est)
        Beta_list = Beta_tolist(Betas)
        W = adj_w(w_est)
        loss = np.zeros(d)
        G_loss = np.zeros([d, d])  # Gradient with respect to w
        G_loss_beta_list = list()  # Gradient with respect to beta
        for j in range(d):
            if loss_type[j] == "gauss":
                M = X @ W  # [n, d]
                R = X[..., j] - M[..., j]
                loss[j] = 0.5 / X.shape[0] * (R**2).sum()  # scalar
                G_loss_j = -1.0 / X.shape[0] * (X.T) @ R  # [d,1]
                G_loss[:, j] = G_loss_j
                G_loss_j_beta = G_loss_j.reshape(-1, 1)
                G_loss_beta_list.append(G_loss_j_beta)

            elif loss_type[j] == "logistic":
                M = X @ W  # [n, d]
                loss[j] = np.sum(
                    1.0
                    / X.shape[0]
                    * (np.logaddexp(0, M[..., j]) - X[..., j] * M[..., j])
                )  # scalar
                G_loss_j = (
                    1.0 / X.shape[0] * X.T @ (sigmoid(M[..., j]) - X[..., j])
                )  # [d, 1]
                G_loss[:, j] = G_loss_j
                G_loss_j_beta = G_loss_j.reshape(-1, 1)
                G_loss_beta_list.append(G_loss_j_beta)
            elif loss_type[j] == "muti-logistic":
                betas = Beta_list[j]  # [d, m]
                M = X @ betas  # [n, m]
                P = softmax(M)  # [n, m]
                # transform x[...,j] to OneHotEncoder
                encoder = OneHotEncoder()
                Y_onehot = encoder.fit_transform(X[..., j].reshape(-1, 1)).toarray()
                loss[j] = np.sum(
                    1.0
                    / X.shape[0]
                    * Y_onehot
                    * (np.log(np.sum(np.exp(M), axis=1, keepdims=True)) - M)
                )
                G_loss_j_beta = (
                    1.0 / X.shape[0] * X.T @ (Y_onehot * P - Y_onehot)
                )  # [d, m] # Gradient with respect to betas
                G_loss_beta_list.append(G_loss_j_beta)

            else:
                raise ValueError("unknown loss type")
        loss = loss.sum()
        print(f"loss={loss}")
        logging.info(f"loss={loss}")
        G_loss_beta = np.concatenate([_ for _ in G_loss_beta_list], axis=1)
        return loss, G_loss, G_loss_beta

    def _h(w_est, beta_est, s=1.0):
        """Compute the log-determinant acyclicity constraint function.
        Referred from:
        https://github.com/kevinsbello/dagma
        """
        W = w_beta_toW(w_est, beta_est)
        M = s * Id - W * W
        h = -la.slogdet(M)[1] + d * np.log(s)
        G_h_W = 2 * W * slin.inv(M).T
        return h, G_h_W

    def _funcbeta(beta):
        """Define the function for optimizing beta parameters."""
        loss, G_loss, G_loss_beta = _loss(w_est=w0, beta_est=beta)
        obj = loss
        G_smooth = G_loss_beta
        g_obj_beta = np.concatenate(
            (G_smooth, -G_smooth), axis=None
        )  # Expand into a one-dimensional array by row
        return obj, g_obj_beta

    def _funcw(w):
        """Define the function for optimizing the weighted adjacency matrix W."""
        loss, G_loss, G_loss_beta = _loss(w_est=w, beta_est=beta0)
        h, G_h = _h(w_est=w, beta_est=beta0)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, -G_smooth + lambda1), axis=None
        )  # Expand into a one-dimensional array by row
        return obj, g_obj

    # Centralize (l2)
    for j in range(d):
        if loss_type[j] == "gauss":
            X[..., j] = X[..., j] - np.mean(X[..., j])

    bnds_w = [
        (0, 0) if i == j else (0, None)
        for _ in range(2)
        for i in range(d)
        for j in range(d)
    ]

    def bnds_wtobeta(bnds):
        """Define the function for optimizing the weighted adjacency matrix W."""
        bnds_w_list = [bnds[i : i + d] for i in range(0, len(bnds), d)]
        res = []
        for tuple_list in bnds_w_list:
            for i, (a, b) in enumerate(tuple_list):
                res.append([(a, b)] * m_vec[i])
        flat_list = [item for sublist in res for item in sublist]
        return flat_list

    bnds_beta = bnds_wtobeta(bnds_w)

    # Optimization
    for _ in range(max_iter):
        # Update beta first, then update w
        print(f"iter:{_}")
        logging.info(f"iter:{_}")
        w_new, h_new = None, None
        while rho < rho_max:
            print(f"rho:{rho}")
            logging.info(f"rho:{rho}")
            w0 = w_est
            sol_beta = sopt.minimize(
                _funcbeta,
                beta_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds_beta,
            )
            # print(f"sol_beta:{sol_beta}")
            beta_new = sol_beta.x
            # print(f"beta:{beta_new}")

            beta0 = beta_new
            sol_w = sopt.minimize(
                _funcw,
                w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bnds_w,
            )
            w_new = sol_w.x
            W_new = adj_w(w_new)
            E_new = slin.expm(W_new * W_new)
            h_new = np.trace(E_new) - d
            print(f"h:{h_new}")
            logging.info(f"h:{h_new}")
            print("######################################################")
            logging.info("######################################################")
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        beta_est, h = beta_new, h_new
        w_est = w_new
        # print(f"beta_est:{beta_new},h:{h_new}")
        alpha += rho * h
        # print(f"alpha:{alpha}")
        if h <= h_tol or rho >= rho_max:
            break

    W_est = w_beta_toW(w_est, beta_est)
    W_est[np.abs(W_est) < w_threshold] = 0

    return W_est


def is_dag(W):
    """Check if the current graph structure is a Directed Acyclic Graph (DAG)."""
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def count_accuracy(B_true, B_est):
    """Compute various directed accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1}

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
        f1: 2 * precision * recall / (precision + recall)
    """

    if not ((B_est == 0) | (B_est == 1)).all():
        raise ValueError("B_est should take value in {0,1}")
    if not is_dag(B_est):
        raise ValueError("B_est should be a DAG")

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # F1-score
    precision = 1 - fdr
    recall = tpr
    f1 = 2 * precision * recall / max((precision + recall), 1)
    print(f"fdr: {fdr}, tpr: {tpr}, fpr: {fpr}, shd: {shd}, nnz: {pred_size}, f1:{f1}")
    logging.info(
        f"fdr: {fdr}, tpr: {tpr}, fpr: {fpr}, shd: {shd}, nnz: {pred_size}, f1:{f1}"
    )
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
    return fdr, tpr, fpr, shd, pred_size, f1


def count_accuracy_und(B_true, B_est):
    """Compute various undirected accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1}

    Returns:
        fdr_und: (false positive) / prediction positive
        tpr_und: (true positive + reverse) / condition positive
        fpr_und: (false positive) / condition negative
        shd_und: undirected extra + undirected missing
        f1_und: 2 * precision_und * recall_und / (precision_und + recall_und)
    """

    # dag
    if not ((B_est == 0) | (B_est == 1)).all():
        raise ValueError("B_est should take value in {0,1}")
    if not is_dag(B_est):
        raise ValueError("B_est should be a DAG")
    d = B_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr_und = float(len(false_pos)) / max(pred_size, 1)
    tpr_und = float(len(reverse) + len(true_pos)) / max(len(cond), 1)
    fpr_und = float(len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd_und = len(extra_lower) + len(missing_lower)
    # F1-score
    precision_und = 1 - fdr_und
    recall_und = tpr_und
    f1_und = 2 * precision_und * recall_und / (precision_und + recall_und)
    print(
        f"fdr_und: {fdr_und}, tpr_und: {tpr_und}, fpr_und: {fpr_und}, shd_und: {shd_und}, nnz: {pred_size}, f1_und:{f1_und}"
    )
    logging.info(
        f"fdr_und: {fdr_und}, tpr_und: {tpr_und}, fpr_und: {fpr_und}, shd_und: {shd_und}, nnz: {pred_size}, f1_und:{f1_und}"
    )
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}
    return fdr_und, tpr_und, fpr_und, shd_und, pred_size, f1_und
