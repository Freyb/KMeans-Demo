import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from random import randint
from math import sqrt, pow


class KMeans:

    def __init__(self):
        self.K_CLUSTERS = 5
        self.COLORS = ['b', 'g', 'y', 'c', 'k', 'm']
        self.RANGES = [60, 80, 80, 170]
        self.DELETE_THRESHOLD = 1
        assert len(self.COLORS) <= 6
        self.iterations = 0
        """data_points = np.array([(1,1),
                      (1,2),
                      (2,2),
                      (5,8),
                      (6,8),
                      (6,9)])"""

        self.data_points = np.loadtxt('dataset.txt')

        self.cluster_points = np.empty((self.K_CLUSTERS, 2)).astype(int)
        for i in range(self.K_CLUSTERS):
            self.cluster_points[i] = (randint(self.RANGES[0], self.RANGES[1]), randint(self.RANGES[2], self.RANGES[3]))

        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.title("K-MEANS ALGORITHM")
        plt.xlabel("Height(Inches)")
        plt.ylabel("Weight(Pounds)")
        plt.axis(self.RANGES)
        self.plotaxes = plt.gca()
        axnext = plt.axes([0.9, 0.9, 0.075, 0.075])
        bnext = Button(axnext, 'Next')
        bnext.on_clicked(self.next_iteration)
        plt.show()

    @staticmethod
    def distance(point1, point2):
        return sqrt(pow(point1[0]-point2[0], 2)+pow(point1[1]-point2[1], 2))

    @staticmethod
    def distance_list(listobj, point):
        return np.sqrt(np.square(listobj[:, 0]-point[0])+np.square(listobj[:, 1]-point[1]))

    @staticmethod
    def distance_matrix(list1, list2):
        matrix = np.empty((len(list2), len(list1)))
        for i in range(len(list1)):
            matrix[:, i] = KMeans.distance_list(list2, list1[i])

        return matrix

    def get_cluster(self, argmin_vector, num):
        cond = np.where(argmin_vector == num)[0]
        sub_data = self.data_points[cond]
        return sub_data

    def calc_mean(self, argmin_vector):
        clusters = np.empty((self.K_CLUSTERS, 2))
        for i in range(self.K_CLUSTERS):
            sub_data = self.get_cluster(argmin_vector, i)
            clusters[i] = np.median(sub_data, 0) if len(sub_data) > 0 else self.cluster_points[i]

        return clusters

    def plot_data(self, argmin_vector):
        self.ax.clear()
        print(self.ax)
        plt.subplot(self.ax)
        plt.title("K-MEANS ALGORITHM")
        plt.xlabel("Height(Inches)")
        plt.ylabel("Weight(Pounds)")
        plt.axis(self.RANGES)
        for j in range(self.K_CLUSTERS):
            clustered_points = self.get_cluster(argmin_vector, j)
            plt.plot(clustered_points[:, 0], clustered_points[:, 1], self.COLORS[j] + 'o')

        plt.plot(self.cluster_points[:, 0], self.cluster_points[:, 1], 'ro')
        plt.draw()

    def run(self):
        print("STARTING POINTS: ", self.cluster_points)
        dm = self.distance_matrix(self.cluster_points, self.data_points)
        am = np.argmin(dm, 1)
        self.cluster_points = self.calc_mean(am)
        self.plot_data(am)

        print("ENDING POINTS: ", self.cluster_points)

    def onclick(self, event):
        if event.inaxes == self.plotaxes:
            if event.button == 1:
                self.data_points = np.append(self.data_points, [[round(event.xdata, 2), round(event.ydata, 2)]], axis=0)

            elif event.button == 3:
                point = [round(event.xdata, 2), round(event.ydata, 2)]
                for i in range(len(self.data_points)):
                    if self.distance(self.data_points[i], point) < self.DELETE_THRESHOLD:
                        self.data_points = np.delete(self.data_points, i, axis=0)
                        break

    def next_iteration(self, event):
        self.iterations += 1
        print("Iteration: ", self.iterations)
        dm = self.distance_matrix(self.cluster_points, self.data_points)
        am = np.argmin(dm, 1)
        # print("AM", am)
        self.cluster_points = self.calc_mean(am)
        self.plot_data(am)
        print("KILEP")


if __name__ == '__main__':
    kmeans = KMeans()
    kmeans.run()
