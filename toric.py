import numpy as np
from scipy.ndimage import laplace

import seaborn as sns
rocket = sns.color_palette("rocket", as_cmap=True)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes

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
        if draw_error:
            ax.add_collection(error_layout(self.error)[1])
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

def error_layout(error: np.ndarray) -> LineCollection:
    """
    Helper function for plotting errors along gridlines.

    Parameters:
    error (np.ndarray): L x L x 2 array representing the error configuration, ∈ ℤ/2ℤ.

    Returns:
    np.ndarray: The line segments representing the errors.
    LineCollection: A collection of line segments representing the errors.
    """
    errors = np.argwhere(error).astype(np.float32)
    x_errors = errors[errors[:,2] == 0][:,:-1]
    y_errors = errors[errors[:,2] == 1][:,:-1]

    x_left = x_errors - 0.5
    x_right = x_left.copy()
    x_right[:,0] += 1
    x_lines = np.stack([x_left, x_right], axis = 1)

    y_left = y_errors - 0.5
    y_right = y_left.copy()
    y_right[:,1] += 1
    y_lines = np.stack([y_left, y_right], axis = 1)

    lines = np.concatenate([x_lines, y_lines], axis = 0)

    return lines, LineCollection(lines, linewidths = 3, colors = 'r')

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