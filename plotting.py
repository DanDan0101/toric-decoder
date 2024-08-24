import numpy as np
import cupy as cp

import seaborn as sns
mako = sns.color_palette("mako", as_cmap=True)

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# from toric import State

def error_layout(x_error: np.ndarray, y_error:np.ndarray, dual: bool = True) -> LineCollection:
    """
    Helper function for plotting errors along gridlines.

    Parameters:
    x_error (np.ndarray): L x L array representing the horizontal errors, ∈ ℤ/2ℤ.
    y_error (np.ndarray): L x L array representing the vertical errors, ∈ ℤ/2ℤ.
    dual (bool): Whether to plot the errors on the dual lattice.

    Returns:
    LineCollection: A collection of line segments representing the errors.
    """

    L = x_error.shape[0]
    x_errors = np.argwhere(x_error)
    y_errors = np.argwhere(y_error)

    # Wrapping for errors on boundaries
    x_boundaries = x_errors[x_errors[:,1] == 0]
    x_boundaries[:,1] += L
    x_errors = np.concatenate([x_errors, x_boundaries], axis = 0)

    y_boundaries = y_errors[y_errors[:,0] == 0]
    y_boundaries[:,0] += L
    y_errors = np.concatenate([y_errors, y_boundaries], axis = 0)

    if dual:
        x_up = x_errors
        x_down = x_errors.copy()
        x_down[:,1] -= 1
        x_lines = np.stack([x_up, x_down], axis = 1)

        y_left = y_errors
        y_right = y_errors.copy()
        y_left[:,0] -= 1
        y_lines = np.stack([y_left, y_right], axis = 1)
    else:
        x_left = x_errors - 0.5
        x_right = x_left.copy()
        x_right[:,0] += 1
        x_lines = np.stack([x_left, x_right], axis = 1)

        y_up = y_errors - 0.5
        y_down = y_up.copy()
        y_up[:,1] += 1
        y_lines = np.stack([y_up, y_down], axis = 1)
    
    lines = np.concatenate([x_lines, y_lines], axis = 0)

    return LineCollection(lines, colors = 'r')

def plot_state(state, dual: bool = True) -> Axes:
    """
    Plots the anyons and errors of the system.

    Parameters:
    state (State): The state of the system.
    dual (bool): Whether to display the errors on the dual lattice.

    Returns:
    Axes: The matplotlib axes object containing the plot.
    """
    
    _, ax = plt.subplots()
    ax.matshow(state.q.T.get(), origin = 'lower', cmap = mako)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.xaxis.tick_bottom()
    ax.add_collection(error_layout(state.x_error.get(), state.y_error.get(), dual = dual))
    return ax

def plot_evolution(history: tuple[cp.ndarray, cp.ndarray, cp.ndarray], shot: int = 0, dual: bool = True) -> animation.FuncAnimation:
    """
    Plot the evolution of the anyon position history.

    Parameters:
    history (tuple): Tuple containing the anyon position, horizontal error, and vertical error history. Each is a T x N x L x L array, ∈ ℤ/2ℤ.
    shot (int): The shot number to plot.
    dual (bool): Whether to display the errors on the dual lattice.

    Returns:
    animation.FuncAnimation: Animation of the anyon evolution.
    """

    q_history = history[0][:,shot,...].get()
    x_error_history = history[1][:,shot,...].get()
    y_error_history = history[2][:,shot,...].get()

    fig, ax = plt.subplots()
    mat = ax.matshow(q_history[0,:,:].T, origin = 'lower', cmap = mako)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Anyon and error evolution')
    ax.xaxis.tick_bottom()
    line = error_layout(x_error_history[0,:,:], y_error_history[0,:,:], dual = dual)
    ax.add_collection(line)

    def update(i):
        mat.set_data(q_history[i,:,:].T)
        line.set_segments(error_layout(x_error_history[i,:,:], y_error_history[i,:,:], dual = dual).get_segments())
        return (mat, line)
    
    return animation.FuncAnimation(fig = fig, func = update, frames = q_history.shape[0], interval = 250, repeat = False)