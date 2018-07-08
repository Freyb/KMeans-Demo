import numpy as np
#from matplotlib import pyplot as plt
from random import randint


def distance(list, point):
    return np.sqrt(np.square(list[:,0]-point[0])+np.square(list[:,1]-point[1]))


def distance_matrix(list1, list2):
    matrix = np.empty((len(list2), len(list1)))
    for i in range(len(list1)):
        matrix[:,i] = distance(list2, list1[i])

    return matrix


def calc_mean(data_points, cluster_points, argmin_vector, K_CLUSTERS):
    clusters = np.empty((K_CLUSTERS,2))
    for i in range(K_CLUSTERS):
        cond = np.where(argmin_vector==i)[0]
        sub_data = data_points[cond]
        clusters[i] = np.mean(sub_data, 0) if len(sub_data)>0 else cluster_points[i]

    return clusters


def main():
    K_CLUSTERS = 2
    ITERATION = 100

    data_points = np.array([(1,1),
              (1,2),
              (2,2),
              (5,8),
              (6,8),
              (6,9)])

    cluster_points = np.empty((K_CLUSTERS, 2)).astype(int)
    for i in range(K_CLUSTERS):
        cluster_points[i] = (randint(0,10), randint(0,10))

    print("STARTING POINTS: ", cluster_points)
    # Algorithm
    for i in range(ITERATION):
        print("ITERATION", i)
        dm = distance_matrix(cluster_points, data_points)
        am = np.argmin(dm, 1)
        print("AM", am)
        cluster_points = calc_mean(data_points, cluster_points, am, K_CLUSTERS)

    print("ENDING POINTS: ", cluster_points)

if __name__ == '__main__':
    main()