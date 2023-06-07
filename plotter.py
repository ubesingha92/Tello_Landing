import matplotlib.pyplot as plt
import time

class RealTimePlotter:
    def __init__(self, title='Real-time Plot'):
        self.x_values = []
        self.y_values = []
        self.h_values = []
        self.timestamps = []
        self.start_time = time.time()
        self.battery_level = None

        plt.ion()  # enable interactive mode
        self.fig, self.ax = plt.subplots(3, 1)  # create a figure and an axis object
        self.fig.suptitle(title)
        for ax in self.ax:
            ax.set_xlabel('Time')
        self.ax[0].set_ylabel('X Error')
        self.ax[1].set_ylabel('Y Error')
        self.ax[2].set_ylabel('H Error')

        # initialize battery level text object
        self.battery_text = self.ax[0].text(0, 1, '', transform=self.ax[0].transAxes)

    def update(self, x_value, y_value, h_value, battery_value):
        self.battery_level = battery_value

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # save the value and timestamp
        self.x_values.append(x_value)
        self.y_values.append(y_value)
        self.h_values.append(h_value)
        self.timestamps.append(elapsed_time)  # relative time

        # remove data older than 20 seconds
        while self.timestamps and self.timestamps[0] < elapsed_time - 20:
            self.timestamps.pop(0)
            self.x_values.pop(0)
            self.y_values.pop(0)
            self.h_values.pop(0)

        # update the plot
        for ax in self.ax:
            ax.clear()
        self.battery_text = self.ax[0].text(0.05, 0.95, 'Battery Level: {}%'.format(self.battery_level), transform=self.ax[0].transAxes, verticalalignment='top')
        self.ax[0].plot(self.timestamps, self.x_values)
        self.ax[0].set_ylim([-150, 150])  # range for x error
        self.ax[1].plot(self.timestamps, self.y_values)
        self.ax[1].set_ylim([-150, 150])  # range for y error
        self.ax[2].plot(self.timestamps, self.h_values)
        self.ax[2].set_ylim([-50, 50])  # range for h error

        plt.pause(0.1)

    def finish(self):
        plt.ioff()  # disable interactive mode
        plt.show()