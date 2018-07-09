import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from random import randint
from math import sqrt, pow


class KMeans:

    def __init__(self):
        self.K_CLUSTERS = 5
        self.ITERATION = 30
        self.COLORS = ['b', 'g', 'y', 'c', 'k', 'm']
        self.RANGES = [60, 80, 80, 170]
        self.DELETE_THRESHOLD = 1
        assert len(self.COLORS) <= 6

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

    def distance(self, point1, point2):
        return sqrt(pow(point1[0]-point2[0], 2)+pow(point1[1]-point2[1], 2))

    def distance_list(self, list, point):
        return np.sqrt(np.square(list[:,0]-point[0])+np.square(list[:,1]-point[1]))

    def distance_matrix(self, list1, list2):
        matrix = np.empty((len(list2), len(list1)))
        for i in range(len(list1)):
            matrix[:,i] = self.distance_list(list2, list1[i])

        return matrix

    def get_cluster(self, argmin_vector, num):
        cond = np.where(argmin_vector == num)[0]
        sub_data = self.data_points[cond]
        return sub_data

    def calc_mean(self, argmin_vector):
        clusters = np.empty((self.K_CLUSTERS,2))
        for i in range(self.K_CLUSTERS):
            sub_data = self.get_cluster(argmin_vector, i)
            clusters[i] = np.median(sub_data, 0) if len(sub_data)>0 else self.cluster_points[i]

        return clusters

    def plot_data(self, argmin_vector):
        plt.clf()
        for j in range(self.K_CLUSTERS):
            clustered_points = self.get_cluster(argmin_vector, j)
            plt.plot(clustered_points[:, 0], clustered_points[:, 1], self.COLORS[j] + 'o')

        plt.plot(self.cluster_points[:, 0], self.cluster_points[:, 1], 'ro')

    def run(self):
        print("STARTING POINTS: ", self.cluster_points)

        # Algorithm
        fig = plt.figure()
        plt.ion()
        plt.show()

        fig.canvas.mpl_connect('button_press_event', self.onclick)

        for i in range(self.ITERATION):
            print("Iteration", i + 1)
            dm = self.distance_matrix(self.cluster_points, self.data_points)
            am = np.argmin(dm, 1)
            # print("AM", am)

            self.cluster_points = self.calc_mean(am)
            self.plot_data(am)
            plt.title("K-MEANS ALGORITHM")
            plt.xlabel("Height(Inches)")
            plt.ylabel("Weight(Pounds)")
            plt.axis(self.RANGES)
            axnext = plt.axes([0.9, 0.9, 0.075, 0.075])
            bnext = Button(axnext, 'Next')
            bnext.on_clicked(self.nextevent)
            plt.draw()
            plt.pause(0.001)
            input("Press [enter] to continue.")

        print("ENDING POINTS: ", self.cluster_points)

    def onclick(self, event):
        """print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))"""
        if event.button == 1:
            print(len(self.data_points))
            self.data_points = np.append(self.data_points, [[round(event.xdata, 2), round(event.ydata, 2)]], axis=0)
            print(len(self.data_points))

        elif event.button == 3:
            point = [round(event.xdata, 2), round(event.ydata, 2)]
            for i in range(len(self.data_points)):
                if self.distance(self.data_points[i], point) < self.DELETE_THRESHOLD:
                    self.data_points = np.delete(self.data_points, i, axis=0)
                    break

    def nextevent(self, event):
        print("ASDASDASD")

if __name__ == '__main__':
    kmeans = KMeans()
    kmeans.run()