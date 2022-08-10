import sys
import numpy as np
import pandas as pd
from argparse import *
np.random.seed(0)



class spkmeans:
    """ main data structure to support the algorithm implementation """

    def __init__(self, n, d, k, goal, data_points):
        """"
        reads the data points from the given input file into an array.
        :param n: number of data points in the input file. (<=1000)
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
        # here we want a function that reads the data points into self.data_points

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
    k = args.k # if k == 0 use the Eigengap Heuristic
    if (not k.isdigit()) or (not (0 <= k < n)):
        invalid_input()
    k = int(k)
    goal = args.goal
    return spkmeans(n, d, k, goal, data_points)
    # כדי לוודא את K אני צריך לדעת מה הn האקטואלי (אני יודע שהוא לא יותר גדול מ1000 אבל זה לא מספיק)


# parse data and call the appropriate spkmeans function based on the goal
# print the result returned from the spkmeans
def invalid_input():
    print("Invalid Input!")
    sys.exit()
def general_error():
    print("An Error Has Occurred")
    sys.exit()

def main():
    parse_input()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        general_error()