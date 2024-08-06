import numpy as np
from utils import *

from scipy.sparse import csc_matrix
from pymatching import Matching

import seaborn as sns
mako = sns.color_palette("mako", as_cmap=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
import matplotlib.animation as animation

from numba import uint8, int32, float32
from numba import jit
from numba.experimental import jitclass

spec = [
    ('L', int32),
    ('N', int32),
    ('q', uint8[:,:]),
    ('error', uint8[:,:,:]),
    ('Φ', float32[:,:]),
]

@jitclass(spec)
class State:
    """
    L (int): lattice size
    N (int): number of anyons
    q (np.ndarray): L x L array representing the anyon field, ∈ ℤ/2ℤ.
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ.
    Φ (np.ndarray): L x L array representing the field, ∈ ℝ.
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
        ax.matshow(self.q.T, origin = 'lower', cmap = mako)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.tick_bottom()
        if draw_error:
            ax.add_collection(error_layout(self.error, dual = True))
        return ax
    
    def add_errors(self, p_error: float) -> None:
        """
        Adds errors to the current state.

        Parameters:
        p_error (float): The probability of an X error occuring per spin.

        Returns:
        None
        """
        y_errors = (np.random.rand(self.L, self.L) < p_error).astype(np.uint8)
        vert_anyons = y_errors ^ roll_2d(y_errors, (-1, 0))

        x_errors = (np.random.rand(self.L, self.L) < p_error).astype(np.uint8)
        horiz_anyons = x_errors ^ roll_2d(x_errors, (0, -1))
        
        self.q ^= vert_anyons ^ horiz_anyons
        self.N = np.sum(self.q)
        self.error ^= np.dstack((x_errors, y_errors))

    def update_field(self, η: float) -> None:
        """
        Update the field state.
        
        Parameters:
        η (float): Smoothing parameter, ∈ (0, 1/2).
        
        Returns:
        None
        """

        self.Φ += η / 4 * laplace_2d(self.Φ)
        self.Φ += self.q
    
    def update_anyon(self) -> None:
        """
        Update the anyon state by moving to the neighbor with largest field value with probability 1/2.

        Returns:
        None
        """

        for x, y in np.argwhere(self.q):
            base = x * self.L + y
            idx = window(self.L) + base
            idx %= self.L ** 2
            neighborhood = self.Φ.take(idx)
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

@jit
def init_state(L: int, p_error: float) -> State:
    """
    Initializes an empty state.

    Parameters:
    L (int): The size of the lattice.
    p_error (float): The probability of an X error occuring per spin.

    Returns:
    State: The initialized state object, with zero field.
    """
    mystate = State(L, 0, np.zeros((L, L), dtype = np.uint8), np.zeros((L, L, 2), dtype = np.uint8))
    mystate.add_errors(p_error)
    return mystate

def pcm(L: int) -> csc_matrix:
    """
    Create a parity-check matrix for the 2D toric code. The rows correspond to plaquettes, 
    and the columns correspond to x bonds and y bonds.

    Parameters:
    L (int): The lattice size.

    Returns:
    csc_matrix: L^2 x 2L^2 sparse array with 4L^2 entries, ∈ ℤ/2ℤ.
    """

    row_ind = np.arange(L**2)

    # x bonds, -y
    col_ind_1 = row_ind.copy()

    # x bonds, +y
    col_ind_2 = (row_ind + 1) % L + row_ind - row_ind % L

    # y bonds, -x
    col_ind_3 = row_ind + L**2

    # y bonds, +x
    col_ind_4 = (row_ind + L) % L**2 + L**2

    row_ind = np.tile(row_ind, 4)
    col_ind = np.concatenate([col_ind_1, col_ind_2, col_ind_3, col_ind_4])
    data = np.ones(4 * L**2, dtype = np.uint8)
    return csc_matrix((data, (row_ind, col_ind)), shape = (L**2, 2 * L**2))

def mwpm(matching: Matching, q: np.ndarray) -> np.ndarray:
    """
    Computes the minimum-weight perfect-matching correction for a given anyon state.

    Parameters:
    matching (Matching): The matching object corresponding to the toric lattice.
    q (np.ndarray): L x L array representing the anyon field, ∈ ℤ/2ℤ.

    Returns:
    np.ndarray: L x L x 2 array representing the correction, ∈ ℤ/2ℤ.
    """

    L = q.shape[0]
    correction = matching.decode(q.flatten())
    x_correction = correction[:L**2].reshape(L,L)
    y_correction = correction[L**2:].reshape(L,L)
    return np.dstack((x_correction, y_correction))

@jit
def logical_error(error: np.ndarray) -> bool:
    """
    Checks if the error configuration corresponds to a logical error.

    Parameters:
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ.

    Returns:
    bool: Whether there is a logical error.
    """

    x_errors = error[:,:,0]
    y_errors = error[:,:,1]

    x_parity = x_errors.sum(axis = 0) % 2
    y_parity = y_errors.sum(axis = 1) % 2

    return x_parity.any() or y_parity.any()

@jit
def decoder_2D(state: State, T: int, c: int, η: float, p_error: float) -> None:
    """
    Run a 2D decoder on a state for T epochs.

    Parameters:
    state (State): The state to decode.
    T (int): Number of epochs to run.
    c (int): Field velocity.
    η (float): Smoothing parameter.
    p_error (float): Probability of an X error occuring per spin, per time step.

    Returns:
    None
    """

    for _ in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        for _ in range(c):
            state.update_field(η)
        state.update_anyon()
        if state.N == 0 and p_error == 0:
            break

def decoder_2D_history(state: State, T: int, c: int, η: float, p_error: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a 2D decoder on a state for T epochs, keeping track of the anyon and error history.

    Parameters:
    state (State): The state to decode.
    T (int): Number of epochs to run.
    c (int): Field velocity.
    η (float): Smoothing parameter.
    p_error (float): Probability of an X error occuring per spin, per time step.

    Returns:
    np.ndarray: T x L x L array representing the anyon position history, ∈ ℤ/2ℤ.
    np.ndarray: T x L x L x 2 array representing the error history, ∈ ℤ/2ℤ.
    """

    q_history = []
    error_history = []
    for _ in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        for _ in range(c):
            state.update_field(η)
        state.update_anyon()
        q_history.append(state.q.copy())
        error_history.append(state.error.copy())
        if state.N == 0 and p_error == 0:
            break
    return np.array(q_history), np.array(error_history)

def error_layout(error: np.ndarray, dual: bool = True) -> LineCollection:
    """
    Helper function for plotting errors along gridlines.

    Parameters:
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ.
    dual (bool): Whether to plot the errors on the dual lattice.

    Returns:
    LineCollection: A collection of line segments representing the errors.
    """

    L = error.shape[0]
    errors = np.argwhere(error).astype(np.float32)
    x_errors = errors[errors[:,2] == 0][:,:-1]
    y_errors = errors[errors[:,2] == 1][:,:-1]

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

def plot_evolution(q_history: np.ndarray, error_history: np.ndarray, dual: bool = True) -> animation.FuncAnimation:
    """
    Plot the evolution of the anyon position history.

    Parameters:
    q_history (np.ndarray): T x L x L array representing the anyon position history, ∈ ℤ/2ℤ.
    error_history (np.ndarray): T x L x L x 2 array representing the error history, ∈ ℤ/2ℤ.
    dual (bool): Whether to display the errors on the dual lattice.

    Returns:
    animation.FuncAnimation: Animation of the anyon evolution.
    """

    fig, ax = plt.subplots()
    mat = ax.matshow(q_history[0,:,:].T, origin = 'lower', cmap = mako)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Anyon and error evolution')
    ax.xaxis.tick_bottom()
    line = error_layout(error_history[0,:,:,:], dual = dual)
    ax.add_collection(line)

    def update(i):
        mat.set_data(q_history[i,:,:].T)
        line.set_segments(error_layout(error_history[i,:,:,:], dual = dual).get_segments())
        return (mat, line)
    
    return animation.FuncAnimation(fig = fig, func = update, frames = q_history.shape[0], interval = 250, repeat = False)