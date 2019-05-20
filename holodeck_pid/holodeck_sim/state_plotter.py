#!/usr/bin/env python
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ViewBox

# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)

class Plotter:
    """
    Class for plotting methods.
    """
    def __init__(self, plotting_frequency=1):
        ''' Initialize the Plotter

            plotting_freq: number of times the update function must be called
                           until the plotter actually outputs the graph.
                           (Can help reduce the slow-down caused by plotting)
        '''
        self.time_window = 15.0
        self.time = 0

        # Able to update plots intermittently for speed
        self.plotting_frequency = plotting_frequency
        self.plot_cnt = 0

        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])
        self.window = pg.GraphicsWindow(title='States')
        self.window.resize(1000,800)


        # Plot default parameters
        self.plots_per_row = 3
        self.default_label_pos = 'left'
        self.auto_adjust_y = True
        # Plot color parameters
        self.distinct_plot_hues = 5 # Number of distinct hues to cycle through
        self.default_plot_hue = 0

        self.plots = {}
        self.curves = {}
        self.curve_colors = {}
        self.states = {}
        self.state_vectors = {}
        self.new_data = False

    def define_state_vector(self, vector_name, state_vector):
        ''' Defines a state vector so measurements can be added in groups

            vector_name (string): name of the vector
            state_vector (list of strings): order of states in the vector

            Note: this does not add states or plots, so plots for the values in
            *state_vector* will need to also be added via the *add_plot* function
        '''
        self.state_vectors[vector_name] = state_vector

    def get_color(self, index):
        ''' Returns incremental plot colors based on index '''
        return pg.intColor(index, hues=self.distinct_plot_hues, minHue=self.default_plot_hue)

    def add_plot_box(self, plot_name, include_legend=False):
        ''' Adds a plot box to the plotting window '''
        if len(self.plots) % self.plots_per_row == 0:
            self.window.nextRow()
        self.plots[plot_name] = self.window.addPlot()
        self.plots[plot_name].setLabel(self.default_label_pos, plot_name)
        if include_legend:
            self.add_legend(plot_name)
        if self.auto_adjust_y:
            state = self.plots[plot_name].getViewBox().getState()
            state["autoVisibleOnly"] = [False, True]
            self.plots[plot_name].getViewBox().setState(state)

    def add_curve(self, plot_name, curve_name, curve_color_idx=0):
        ''' Adds a curve to the specified plot

            plot_name: Name of the plot the curve will be added to
            curve_name: Name the curve will be referred by
            curve_color_idx: index of the curve in the given plot
                       (i.e. 0 if it's the first curve, 1 if it's the second, etc.)
                       Used to determine the curve color with *get_color* function
        '''
        curve_color = self.get_color(curve_color_idx)
        self.curves[curve_name] = self.plots[plot_name].plot(name=curve_name)
        self.curve_colors[curve_name] = curve_color
        self.states[curve_name] = []

    def add_plot(self, curve_names, include_legend=False):
        ''' Adds a state and the necessary plot, curves, and data lists

            curve_names: name(s) of the state(s) to be plotted in the same plot window
                         (e.g. ['x', 'x_truth'] or ['x', 'x_command'])
        '''
        # Check if the input is given as a string or a list
        if type(curve_names) == str:
            # There is only a single curve name, represented by a string, not a list
            self.add_plot_box(curve_names, include_legend)
            self.add_curve(curve_names, curve_names)
        elif type(curve_names) == list:
            # Initialize plot
            plot_name = curve_names[0]
            self.add_plot_box(plot_name, include_legend)

            # Add each curve to the plot
            curve_color_idx = 0
            for curve_name in curve_names:
                self.add_curve(plot_name, curve_name, curve_color_idx)
                curve_color_idx += 1
        else:
            print("ERROR: Invalid type for 'curve_names' input. Please use a string or list")

    def add_legend(self, plot_name):
        self.plots[plot_name].addLegend(size=(1,1), offset=(1,1))

    def add_measurement(self, state_name, state_val, time):
        '''Adds a measurement for the given state

            state_name (string): name of the state
            state_val (number): value to be added for the state
            time (number): time (in seconds) of the measurement
        '''
        self.states[state_name].append([time, state_val])
        self.new_data = True
        if time > self.time:
            self.time = time # Keep track of the latest data point

    def add_vector_measurement(self, vector_name, vector_values, time):
        '''Adds a group of measurements in vector form

            vector_name (string): name given the vector through the *define_state_vector*
                                  function
            vector_values (list of numbers): values for each of the states in the
                          order defined in the *define_state_vector* function
            time: time stamp for the values in the vector

        '''
        state_index = 0
        if len(vector_values) != len(self.state_vectors[vector_name]):
            print("ERROR: State vector length mismatch. \
                          State vector '{0}' has length {1}".format(vector_name, len(vector_values)))
        for state in self.state_vectors[vector_name]:
            self.add_measurement(state, vector_values[state_index], time)
            state_index += 1

    # Update the plots with the current data
    def update_plots(self):
        '''Updates the plots (according to plotting frequency defined in initialization) '''
        self.plot_cnt += 1
        if self.new_data and (self.plot_cnt % self.plotting_frequency == 0):

            for curve in self.curves:
                data = self.states[curve]
                # Reshape the data
                data = np.reshape(data, np.shape(data))
                time_array = data[:,0]
                values_array = data[:,1]
                self.curves[curve].setData(time_array, values_array, pen=self.curve_colors[curve])

            x_min = max(self.time - self.time_window, 0)
            x_max = self.time
            for plot in self.plots:
                self.plots[plot].setXRange(x_min, x_max)
                self.plots[plot].enableAutoRange(axis=ViewBox.YAxis)

            self.new_data = False

        # update the plotted data
        self.app.processEvents()
