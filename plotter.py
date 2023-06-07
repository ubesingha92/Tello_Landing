import matplotlib.pyplot as plt
import time

class RealTimePlotter:
    def __init__(self, value_range=None, title='Real-time Plot'):
        self.values = []
        self.timestamps = []
        self.start_time = time.time()
        self.value_range = value_range

        plt.ion()  # enable interactive mode
        self.fig, self.ax = plt.subplots()  # create a figure and an axis object
        self.ax.set_title(title)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        
    def update(self, value):
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # save the value and timestamp
        self.values.append(value)
        self.timestamps.append(elapsed_time)  # relative time

        # remove data older than 20 seconds
        while self.timestamps and self.timestamps[0] < elapsed_time - 20:
            self.timestamps.pop(0)
            self.values.pop(0)
        
        # update the plot
        self.ax.clear()
        self.ax.plot(self.timestamps, self.values)
        
        if self.value_range is not None:
            self.ax.set_ylim(self.value_range)  # set the y-limits to the value range

        plt.pause(0.1)

    def finish(self):
        plt.ioff()  # disable interactive mode
        plt.show()
