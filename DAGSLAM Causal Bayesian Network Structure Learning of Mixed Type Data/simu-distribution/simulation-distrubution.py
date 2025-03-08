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
        filename="simu-distribution-sem_type.log",
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

    def simulate_parameter(B, alpha=1):
        """Simulate SEM parameters for a DAG.

        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            alpha (float): Scaling factor for the weights (default is 1).

        Returns:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
        """
        w_ranges = ((-2.0 * alpha, -0.5 * alpha), (0.5 * alpha, 2.0 * alpha))
        W = np.zeros(B.shape)
        S = np.random.randint(len(w_ranges), size=B.shape)
        for i, (low, high) in enumerate(w_ranges):
            U = np.random.uniform(low=low, high=high, size=B.shape)
            W += B * (S == i) * U
        return W

    def simulate_linear_sem(W, n, Sem_Type, noise_scale=None):
        """Simulate samples from linear SEM with specified type of noise.

        For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

        Args:
            W (np.ndarray): [d, d] weighted adj matrix of DAG
            n (int): num of samples, n=inf mimics population risk
            Sem_type (list): gauss, exp, gumbel, logistic
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones

        Returns:
            X (np.ndarray): [n, d] sample matrix
        """

        def _simulate_single_equation(X, w, sem_type, scale):
            """X: [n, num of parents], w: [num of parents], x: [n]"""
            """
            Simulate a single equation for a given SEM type.

            Args:
                X (np.ndarray): [n, num of parents] matrix of parent variables.
                w (np.ndarray): [num of parents] coefficients for the equation.
                sem_type (str): Type of SEM ('gauss', 'exp', 'gumbel', 'logistic').
                scale (float): Scale parameter for the noise.

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
            X[:, j] = _simulate_single_equation(
                X[:, parents], W[parents, j], type_vec[j], scale_vec[j]
            )
        return X

    default_save_path = "."
    import os

    graph_type = "ER"
    d = 20
    s0 = d

    for n in [1000]:
        for distribution in ["gauss", "exp", "gumbel"]:
            filenameB = os.path.join(
                default_save_path, f"B_true_n{n}_{distribution}.csv"
            )
            filenameW = os.path.join(
                default_save_path, f"W_true_n{n}_{distribution}.csv"
            )

            # Choose to either generate a new B_true or use the existing B_true from the CSV file
            # B_true = simulate_dag(d, s0, graph_type)
            # np.savetxt(filenameB, B_true, delimiter=",")
            B_true = pd.read_csv(filenameB, header=None)  # Use the existing one
            B_true = B_true.values

            # Choose to either generate a new W_true or use the existing W_true from the CSV file
            # W_true = simulate_parameter(B_true)
            # np.savetxt(filenameW, W_true, delimiter=",")
            W_true = pd.read_csv(filenameW, header=None)  # Use the existing one
            W_true = W_true.values
            random.seed(42)
            print(f"distribution={distribution}")
            logging.info(f"distribution={distribution}")
            sem_type = ["logistic"] + [distribution] * (d - 1)
            random.shuffle(sem_type)
            print(f"sem_type:{sem_type}")
            logging.info(f"sem_type:{sem_type}")

            for i in range(10):  # 10 replications
                filenameX = os.path.join(
                    default_save_path, f"X_n{n}_{distribution}_i{i}.csv"
                )
                X = simulate_linear_sem(W_true, n, sem_type)
                np.savetxt(filenameX, X, delimiter=",")


main()
