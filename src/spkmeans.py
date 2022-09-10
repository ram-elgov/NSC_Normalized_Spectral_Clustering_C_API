import math
import sys
import numpy as np
import pandas as pd
from argparse import *
import finalmodule

np.random.seed(0)
MAX_ITER = 300


def print_output_centroids(km):
    """
    outputs the final centroids calculated by kmeans.c
    """
    for centroid in km.output:
        print(','.join(["%.4f" % coordinate for coordinate in centroid]))


def print_centroid_indices(km):
    """
    outputs the final centroids indexes.
    """
    print(','.join([f"{int(i)}" for i in km.centroids_indices]))


class SpectralClustering:
    """ main data structure to support the algorithm implementation """

    def __init__(self, n, d, k, goal, data_points, max_iter):
        """"
        reads the data data_points from the given input file into an array.
        :param n: number of data data_points in the input file. (<=1000)
        :param d: the dimension of each data point. 
        :param k: Number of required clusters (0 <= k < n)
        :param goal: Can get the following values:
                    spk: Perform full spectral kmeans as described in 1.
                    wam: Calculate and output the Weighted Adjacency Matrix as described in 1.1.1.
                    ddg: Calculate and output the Diagonal Degree Matrix as described in 1.1.2.
                    lnorm: Calculate and output the Normalized Graph Laplacian as described in 1.1.3.
                    jacobi: Calculate and output the eigenvalues and eigenvectors as described in 1.2.1.
        """""
        self.data_points = data_points
        self.n = n
        self.d = d
        self.k = k
        self.goal = goal
        self.max_iter = max_iter


def parse_input():
    """
    parse the input arguments and validates them.
    :return: spkmeans object initialized with respect
     to the given user input
    """
    parser = ArgumentParser()
    parser.add_argument("k")
    parser.add_argument("goal", type=str)
    parser.add_argument("file_name", type=str)
    args = parser.parse_args()
    file_name = args.file_name
    data_points = pd.read_csv(file_name, header=None)
    n = data_points.shape[0]
    d = data_points.shape[1]
    k = args.k  # if k == 0 use the Eigengap Heuristic
    if not k.isdigit():
        invalid_input()
    k = int(k)
    if not 0 <= k < n:
        invalid_input()
    goal = args.goal
    # data_point is flattem and converted to a list to match C/API input
    return SpectralClustering(n, d, k, goal, data_points.to_numpy().flatten().tolist(), MAX_ITER)


# parse data and call the appropriate spkmeans function based on the goal
# print the result returned from the spkmeans
def invalid_input():
    print("Invalid Input!")
    sys.exit()


def general_error():
    print("An Error Has Occurred")
    sys.exit()


def print_matrix(matrix, n, d):
    for i in range(n):
        for j in range(d):
            if j != d - 1:
                print('%.4f,' % matrix[i * d + j], end="")
            else:
                print('%.4f' % matrix[i * d + j], end="")
        print("")


class KMeans:
    """
    data structure to store the required data for the algorithm.
    """

    def __init__(self, k, epsilon, file_name_1=None, file_name_2=None, max_iter=MAX_ITER, data_points=None):
        self.k = k  # number of clusters
        self.max_iter = max_iter  # maximum number of iteration for the algorithm
        self.file_name_1 = file_name_1  # an input file with valid format of data points (text file)
        self.file_name_2 = file_name_2  # an input file to save the results into (text file)
        self.epsilon = epsilon  # the accepted error
        self.data_points = pd.array([])
        self.initialize_data_points(data_points)
        self.number_of_rows = self.data_points.shape[0]
        self.number_of_cols = self.data_points.shape[1]
        if data_points is None:
            self.data_points = self.data_points.to_numpy()
        if not (1 < self.k < self.number_of_rows):
            invalid_input()
        self.centroids = pd.array([])
        self.centroids_indices = []
        self.output = None
        self.D = np.array([])
        self.P = np.array([])

    def initialize_data_points(self, data_points=None):
        """
        reads and merge the two input files
        :return:
        """
        if data_points is None:
            input_1 = pd.read_csv(self.file_name_1, header=None)
            input_2 = pd.read_csv(self.file_name_2, header=None)
            self.data_points = pd.merge(input_1, input_2, how="inner", left_on=input_1.columns[0],
                                        right_on=input_2.columns[0])
            self.data_points.sort_values(by=self.data_points.columns[0], inplace=True)
            self.data_points.drop(self.data_points.columns[0], axis=1, inplace=True)
            # look https://moodle.tau.ac.il/mod/forum/discuss.php?d=104697 in the forum
        else:
            self.data_points = data_points

    def find_min_distance(self, data_point):
        """
        a function to compute the minimal distance between a given data frame to the current existing
        centroids. assumes centroids is not empty.
        param data_point: a given data point to compute the minimal distance for.
        :return: the minimal distance between the input and the centroids.
        """
        m = math.inf
        for i in range(self.centroids.shape[0]):
            m = min(m, np.sum(np.power(np.subtract(data_point, self.centroids[i]), 2)))
        return m

    def k_means_pp(self):
        """
        an implementation of the kmeans++ algorithm to generate initial centroids
        for the use of a kmeans clustering algorithm implementation.
        assumes initialize_centroids() was already called.
        :return: a float type data frame that contains the randomly chosen centroids.
        """
        np.random.seed(0)
        miu1_index = np.random.choice(range(self.number_of_rows))
        self.centroids = np.array([self.data_points[miu1_index]])
        self.centroids_indices.append(miu1_index)
        for i in range(1, self.k):
            self.D = np.array([self.find_min_distance(self.data_points[curr])
                               for curr in range(self.number_of_rows)])
            sum_d = np.sum(self.D)
            self.P = np.array([d / sum_d for d in self.D])
            random_centroid_i = np.random.choice(range(self.number_of_rows), p=self.P)
            self.centroids = np.append(self.centroids,
                                       np.array([self.data_points[random_centroid_i]]), axis=0)
            self.centroids_indices.append(random_centroid_i)


def main():
    spk = parse_input()
    if spk.goal == 'spk':
        tuple_t_k = finalmodule.fit(spk.data_points, spk.n, spk.d, spk.k)
        t = tuple_t_k[0]
        spk.k = tuple_t_k[1]
        t = np.reshape(t, (spk.n, spk.k))
        kmeans = KMeans(spk.k, 0, file_name_1=None, file_name_2=None, data_points=t)
        kmeans.k_means_pp()
        print_centroid_indices(kmeans)
        kmeans.output = finalmodule.fit_kmeans(
            spk.n, spk.k, spk.max_iter, spk.k, kmeans.epsilon, kmeans.centroids.tolist(), kmeans.data_points.tolist())
        print_output_centroids(kmeans)
    elif spk.goal == "wam":
        print_matrix(finalmodule.compute_wam(spk.data_points, spk.n, spk.d), spk.n, spk.n)
    elif spk.goal == "ddg":
        print_matrix(finalmodule.compute_ddg(spk.data_points, spk.n, spk.d), spk.n, spk.n)
    elif spk.goal == "lnorm":
        print_matrix(finalmodule.compute_lnorm(spk.data_points, spk.n, spk.d), spk.n, spk.n)
    elif spk.goal == "jacobi":
        print_matrix(finalmodule.compute_jacobi(spk.data_points, spk.n, spk.d), spk.n + 1, spk.n)
    else:
        invalid_input()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(str(e) + "\n")
        general_error()
