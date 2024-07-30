import numpy as np
from scipy.ndimage import laplace

import seaborn as sns
rocket = sns.color_palette("rocket", as_cmap=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
import matplotlib.animation as animation

from typing import Union

WINDOW = np.array([-1, 0, 1])

class State:
    """
    L (int): lattice size
    N (int): number of anyons
    q (np.ndarray): L x L array representing the anyon field, ∈ ℤ/2ℤ
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ
    Φ (np.ndarray): L x L array representing the field, ∈ ℝ
    """

    def __init__(self, L: int, N: int, q: np.ndarray, error: np.ndarray):
        self.L = L
        self.N = N
        self.q = q
        self.error = error
        self.Φ = np.zeros((L, L), dtype = np.float32)

    def draw(self, draw_error: bool = True) -> Axes:
        """
        Plot the state.

        Parameters:
        draw_error (bool): Whether to plot the errors.

        Returns:
        Axes: The matplotlib axes object containing the plot.
        """

        _, ax = plt.subplots()
        ax.matshow(self.q.T, origin = 'lower', cmap = rocket)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.tick_bottom()
        if draw_error:
            ax.add_collection(error_layout(self.error))
        return ax
    
    def update_field(self, η: float) -> None:
        """
        Update the field state.
        
        Parameters:
        η (float): Smoothing parameter, ∈ (0, 1/2).
        
        Returns:
        None
        """

        self.Φ += η / 4 * laplace(self.Φ, mode='wrap')
        self.Φ += self.q
    
    def update_anyon(self) -> None:
        """
        Update the anyon state by moving to the neighbor with largest field value with probability 1/2.

        Returns:
        None
        """

        for x, y in np.argwhere(self.q):
            idx = x * self.L + y
            shifts = np.add.outer(WINDOW * self.L, WINDOW).flatten()
            neighborhood = self.Φ.take(shifts + idx, mode = 'wrap')
            neighborhood = neighborhood[1::2]
            direction = neighborhood.argmax()
            if np.random.rand() < 0.5:
                self.q[x, y] ^= 1
                if direction == 0: # -x
                    self.q[(x - 1) % self.L, y] ^= 1
                    self.error[x, y, 1] ^= 1
                elif direction == 1: # -y
                    self.q[x, (y - 1) % self.L] ^= 1
                    self.error[x, y, 0] ^= 1
                elif direction == 2: # +y
                    self.q[x, (y + 1) % self.L] ^= 1
                    self.error[x, (y + 1) % self.L, 0] ^= 1
                elif direction == 3: # +x
                    self.q[(x + 1) % self.L, y] ^= 1
                    self.error[(x + 1) % self.L, y, 1] ^= 1
                else:
                    raise ValueError("Invalid direction")
        self.N = np.sum(self.q)

def error_layout(error: np.ndarray, dual: bool = True) -> LineCollection:
    """
    Helper function for plotting errors along gridlines.

    Parameters:
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ.
    dual (bool): Whether to plot the errors on the dual lattice.

    Returns:
    LineCollection: A collection of line segments representing the errors.
    """

    errors = np.argwhere(error).astype(np.float32)
    x_errors = errors[errors[:,2] == 0][:,:-1]
    y_errors = errors[errors[:,2] == 1][:,:-1]

    if dual:
        x_left = x_errors.copy()
        x_right = x_errors.copy()
        x_right[:,1] -= 1

        y_left = y_errors.copy()
        y_right = y_errors.copy()
        y_left[:,0] -= 1
    else:
        x_left = x_errors - 0.5
        x_right = x_left.copy()
        x_right[:,0] += 1

        y_left = y_errors - 0.5
        y_right = y_left.copy()
        y_right[:,1] += 1
    
    x_lines = np.stack([x_left, x_right], axis = 1)
    y_lines = np.stack([y_left, y_right], axis = 1)
    lines = np.concatenate([x_lines, y_lines], axis = 0)

    return LineCollection(lines, colors = 'r')

def init_state(L: int, p_error: float) -> State:
    """
    Initializes a state with a random distribution of anyons.

    Parameters:
    L (int): The size of the lattice.
    p_error (float): The probability of an X error occuring per spin.

    Returns:
    State: The initialized state object, with zero field.
    """

    y_errors = (np.random.rand(L, L) < p_error).astype(np.int32)
    vert_anyons = y_errors ^ np.roll(y_errors, -1, axis=0)

    x_errors = (np.random.rand(L, L) < p_error).astype(np.int32)
    horiz_anyons = x_errors ^ np.roll(x_errors, -1, axis=1)
    
    q = vert_anyons ^ horiz_anyons
    N = np.sum(q)
    return State(L, N, q, np.stack([x_errors, y_errors], axis = 2))

def plot_evolution(q_history: np.ndarray, error_history: np.ndarray, trail: bool, dual: bool = True) -> animation.FuncAnimation:
    """
    Plot the evolution of the anyon position history.

    Parameters:
    q_history (np.ndarray): T x L x L array representing the anyon position history.
    error_history (np.ndarray): T x L x L x 2 array representing the error history.
    trail (bool): Whether to display a trail behind the anyon as it moves.
    dual (bool): Whether to display the errors on the dual lattice.

    Returns:
    animation.FuncAnimation: Animation of the anyon evolution.
    """

    fig, ax = plt.subplots()
    mat = ax.matshow(q_history[0,:,:].T, origin = 'lower', cmap = rocket)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Anyon and error evolution')
    ax.xaxis.tick_bottom()
    line = error_layout(error_history[0,:,:,:], dual = dual)
    ax.add_collection(line)

    def update(i):
        if trail and i > 0:
            previous = q_history[i-1::-1,:,:]
            weights = 2 ** np.arange(-1, -i-1, -1, dtype = np.float32)
            history = np.tensordot(previous, weights, axes = (0, 0))
            mat.set_data(np.maximum(q_history[i,:,:], history).T)
        else:
            mat.set_data(q_history[i,:,:].T)
        line.set_segments(error_layout(error_history[i,:,:,:], dual = dual).get_segments())
        return (mat, line)
    
    return animation.FuncAnimation(fig = fig, func = update, frames = q_history.shape[0], interval = 250)

def decoder_2D(state: State, T: int, c: int, η: float, history: bool) -> Union[None, tuple[np.ndarray, np.ndarray]]:
    """
    Run a 2D decoder on a state for T epochs.

    Parameters:
    state (State): The state to decode.
    T (int): Number of epochs to run.
    c (int): Field velocity.
    η (float): Smoothing parameter.
    history (bool): Whether to return the history of anyon positions and errors.

    Returns:
    np.ndarray: T x L x L array representing the anyon position history.
    np.ndarray: T x L x L x 2 array representing the error history.
    """

    q_history = []
    error_history = []
    for _ in range(T):
        for _ in range(c):
            state.update_field(η)
        state.update_anyon()
        # Add some errors
        if history:
            q_history.append(state.q.copy())
            error_history.append(state.error.copy())
        if state.N == 0:
            break
    if history:
        return np.array(q_history), np.array(error_history)

def logical_error(error: np.ndarray) -> bool:
    """
    Checks if the error configuration corresponds to a logical error.

    Parameters:
    error (np.ndarray): L x L x 2 array representing the error configuration.

    Returns:
    bool: Whether there is a logical error.
    """

    x_errors = error[:,:,0]
    y_errors = error[:,:,1]

    x_parity = x_errors.sum(axis = 0) % 2
    y_parity = y_errors.sum(axis = 1) % 2

    return x_parity.any() or y_parity.any()