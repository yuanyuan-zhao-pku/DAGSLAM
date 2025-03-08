"""  
This algorithm is an extension of the NOTEARS algorithm.  
The original NOTEARS algorithm was developed by Xun Zheng, et al.  
This implementation is authored by Yuanyuan Zhao.  
"""


def main():
    """
    Main function to generate a a series of simulated dataset X and the corresponding true weighted adjacency matrix W_true.

    This function performs the following tasks:
    1. Sets up logging for tracking the simulation process.
    2. Defines parameters for the simulation, including graph type, number of nodes, and edges.
    3. Simulates a true DAG and its corresponding parameters.
    4. Generates samples from a linear SEM based on the simulated DAG and parameters.
    5. Saves the generated adjacency matrices and samples to CSV files for further analysis.
    """
    import numpy as np
    import pandas as pd
    from scipy.special import expit as sigmoid
    import igraph as ig
    import random
    import logging

    logging.basicConfig(
        filename="simu-catnum.log",
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    def is_dag(W):
        """
        Check if the given adjacency matrix represents a Directed Acyclic Graph (DAG).

        Args:
            W (np.ndarray): [d, d] binary adjacency matrix.

        Returns:
            bool: True if W represents a DAG, False otherwise.
        """
        G = ig.Graph.Weighted_Adjacency(W.tolist())
        return G.is_dag()

    def simulate_dag(d, s0, graph_type):
        """Simulate random DAG with some expected number of edges.

        Args:
            d (int): num of nodes
            s0 (int): expected num of edges
            graph_type (str): ER, SF, BP

        Returns:
            B (np.ndarray): [d, d] binary adj matrix of DAG
        """

        def _random_permutation(M):
            """Generate a random permutation of the matrix M."""
            P = np.random.permutation(np.eye(M.shape[0]))
            return P.T @ M @ P

        def _random_acyclic_orientation(B_und):
            """Generate a random acyclic orientation of the undirected graph."""
            return np.tril(_random_permutation(B_und), k=-1)

        def _graph_to_adjmat(G):
            """Convert a graph to its adjacency matrix."""
            return np.array(G.get_adjacency().data)

        if graph_type == "ER":
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == "SF":
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError("unknown graph type")
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        return B_perm

    def simulate_parameter(B, m, alpha=1):
        """Simulate SEM parameters for a DAG.

        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            m (int):  The number of categories for multi-categorical variables
            alpha (float): Scaling factor for the weights (default is 1).

        Returns:
            W (list): A list of [d, d] weighted adjacency matrices for the SEM.
        """
        w_ranges = ((-2.0 * alpha, -0.5 * alpha), (0.5 * alpha, 2.0 * alpha))
        W = np.zeros(B.shape)

        S = np.random.randint(len(w_ranges), size=B.shape)
        W = list()
        for i in range(m):
            temp = np.zeros(B.shape)
            for i, (low, high) in enumerate(w_ranges):
                U = np.random.uniform(low=low, high=high, size=B.shape)
                temp += B * (S == i) * U
            W.append(temp)
        return W

    def simulate_linear_sem(
        n, Sem_Type, noise_scale=None, W=None, Wlist=None, m_vec=None
    ):
        """Simulate samples from linear SEM with specified type of noise.

        Args:
            n (int): Number of samples to simulate.
            Sem_Type (list): List of noise types for each variable.
            noise_scale (np.ndarray): Scale parameter of additive noise (default is None).
            W (np.ndarray): [d, d] weighted adjacency matrix of the DAG (default is None).
            Wlist (list): List of weighted adjacency matrices (default is None).
            m_vec (list): Number of categories for multi-categorical variables (default is None).

        Returns:
            X (np.ndarray): [n, d] matrix of simulated samples.
        """
        if W == None:
            W = Wlist[1]

        def _simulate_single_equation(X, sem_type, scale, m, w, wlist=None):
            """
            Simulate a single equation for a given SEM type.

            Args:
                X (np.ndarray): [n, num of parents] matrix of parent variables.
                sem_type (str): Type of SEM ('gauss', 'exp', 'gumbel', 'logistic', 'muti-logistic').
                scale (float): Scale parameter for the noise.
                m (int): Number of categories for multi-categorical variables.
                w (np.ndarray): [num of parents] coefficients for the equation.
                wlist (np.ndarray): [m, num of parent] natrix of coefficients for multi-categorical variables (default is None).

            Returns:
                x(np.ndarray): [n] simulated values for the dependent variable.
            """
            if sem_type == "gauss":
                z = np.random.normal(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "exp":
                z = np.random.exponential(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "gumbel":
                z = np.random.gumbel(scale=scale, size=n)
                x = X @ w + z
            elif sem_type == "logistic":
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
            elif sem_type == "muti-logistic":
                probabilities = np.exp(np.dot(X, wlist.T))
                probabilities /= np.sum(probabilities, axis=1, keepdims=True)
                x = np.array(
                    [np.random.choice(range(m), p=prob) for prob in probabilities]
                )
            else:
                raise ValueError("unknown sem type")
            return x

        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError("noise scale must be a scalar or has length d")
            scale_vec = noise_scale
        if not is_dag(W):
            raise ValueError("W must be a DAG")

        if Sem_Type is None:
            type_vec = ["gauss"] * d
        elif len(Sem_Type) == 1:
            type_vec = Sem_Type * d
        else:
            if len(Sem_Type) != d:
                raise ValueError("Sem_Type must be a scalar or has length d")
            type_vec = Sem_Type

        G = ig.Graph.Weighted_Adjacency(W.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            w = W[parents, j]
            if Wlist != None:
                wlist = np.array(list(W[parents, j] for W in Wlist))
            else:
                wlist = None

            X[:, j] = _simulate_single_equation(
                X=X[:, parents],
                sem_type=type_vec[j],
                scale=scale_vec[j],
                m=m_vec[j],
                w=w,
                wlist=wlist,
            )
        return X

    def sem_type_to_m_vec(sem_type, m):
        """
        Convert SEM types to their corresponding category counts.

        Args:
            sem_type (list): List of SEM types for each variable.
            m (int): Number of categories for multi-categorical variables.

        Returns:
            list: A list indicating the number of categories for each SEM type.
        """
        m_vec = [m if s == "muti-logistic" else 1 for s in sem_type]
        return m_vec

    default_save_path = "."
    import os

    graph_type = "ER"
    d = 20
    s0 = d

    for n in [1000]:
        for m in [3, 4]:
            filenameB = os.path.join(default_save_path, f"B_true_n{n}_m{m}.csv")

            # Choose to either generate a new B_true or use the existing B_true from the CSV file
            # B_true = simulate_dag(d, s0, graph_type)
            # np.savetxt(filenameB, B_true, delimiter=",")
            B_true = pd.read_csv(filenameB, header=None)  # Use the existing one
            B_true = B_true.values

            W_true_list = simulate_parameter(B_true, m=m)
            import csv

            for i, arr in enumerate(W_true_list):
                filenameW = os.path.join(default_save_path, f"W_true_n{n}_m{m}_{i}.csv")
                np.savetxt(filenameW, arr, delimiter=",")

            random.seed(42)
            sem_type = ["muti-logistic"] + ["gauss"] * (d - 1)
            random.shuffle(sem_type)
            print(f"m={m}")
            logging.info(f"m={m}")
            print(f"sem_type:{sem_type}")
            logging.info(f"sem_type:{sem_type}")
            m_vec = sem_type_to_m_vec(sem_type=sem_type, m=m)
            print(f"m_vec={m_vec}")
            logging.info(f"m_vec={m_vec}")

            W_true = W_true_list[1]
            for j in range(d):
                if m_vec[j] != 1:
                    W_true[:, j] = np.sum([W[:, j] ** 2 for W in W_true_list], axis=0)
            filenameW = os.path.join(default_save_path, f"W_true_n{n}_m{m}.csv")
            np.savetxt(filenameW, W_true, delimiter=",")

            for i in range(10):  # 10 replications
                filenameX = os.path.join(default_save_path, f"X_n{n}_m{m}_i{i}.csv")
                X = simulate_linear_sem(
                    n=n, Sem_Type=sem_type, Wlist=W_true_list, m_vec=m_vec
                )
                np.savetxt(filenameX, X, delimiter=",")


main()
