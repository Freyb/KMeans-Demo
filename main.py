import numpy as np
from matplotlib import pyplot as plt
from random import randint


def distance(list, point):
    return np.sqrt(np.square(list[:,0]-point[0])+np.square(list[:,1]-point[1]))


def distance_matrix(list1, list2):
    matrix = np.empty((len(list2), len(list1)))
    for i in range(len(list1)):
        matrix[:,i] = distance(list2, list1[i])

    return matrix


def get_cluster(data_points, argmin_vector, num):
    cond = np.where(argmin_vector == num)[0]
    sub_data = data_points[cond]
    return sub_data


def calc_mean(data_points, cluster_points, argmin_vector, K_CLUSTERS):
    clusters = np.empty((K_CLUSTERS,2))
    for i in range(K_CLUSTERS):
        sub_data = get_cluster(data_points, argmin_vector, i)
        clusters[i] = np.median(sub_data, 0) if len(sub_data)>0 else cluster_points[i]

    return clusters


def plot_data(data_points, cluster_points, argmin_vector, K_CLUSTERS, COLORS):
    plt.clf()
    for j in range(K_CLUSTERS):
        clustered_points = get_cluster(data_points, argmin_vector, j)
        plt.plot(clustered_points[:, 0], clustered_points[:, 1], COLORS[j] + 'o')

    plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'ro')


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))


def main():
    K_CLUSTERS = 5
    ITERATION = 30
    COLORS = ['b', 'g', 'y', 'c', 'k', 'm']
    RANGES = [60, 80, 80, 170]
    assert len(COLORS)<=6

    """data_points = np.array([(1,1),
              (1,2),
              (2,2),
              (5,8),
              (6,8),
              (6,9)])"""

    data_points = np.loadtxt('dataset.txt')

    cluster_points = np.empty((K_CLUSTERS, 2)).astype(int)
    for i in range(K_CLUSTERS):
        cluster_points[i] = (randint(RANGES[0], RANGES[1]), randint(RANGES[2], RANGES[3]))

    print("STARTING POINTS: ", cluster_points)

    # Algorithm
    fig = plt.figure()
    plt.ion()
    plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    for i in range(ITERATION):
        print("Iteration", i+1)
        dm = distance_matrix(cluster_points, data_points)
        am = np.argmin(dm, 1)
        # print("AM", am)

        plot_data(data_points, cluster_points, am, K_CLUSTERS, COLORS)
        plt.title("K-MEANS ALGORITHM")
        plt.xlabel("Height(Inches)")
        plt.ylabel("Weight(Pounds)")
        plt.axis(RANGES)
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

        cluster_points = calc_mean(data_points, cluster_points, am, K_CLUSTERS)

    print("ENDING POINTS: ", cluster_points)


if __name__ == '__main__':
    main()