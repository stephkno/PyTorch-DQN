import pyimplot
import numpy as np

if pyimplot.begin_plot("Test Plot"):
    xs = np.arange(10, dtype=np.float64)
    ys = np.random.rand(10)
    pyimplot.plot_line("Random", xs, ys)
    pyimplot.end_plot()
