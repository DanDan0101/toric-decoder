import numpy as np
from scipy.sparse import csc_matrix

from pymatching import Matching

import cupy as cp
from cupyx.scipy.ndimage import correlate1d, correlate
OONO_PURI = cp.array([
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    [[1, 2, 1], [2, -12, 2], [1, 2, 1]],
    [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
])
from stride_tricks import sliding_window_view

from plotting import plot_state
from matplotlib.axes import Axes

class State:
    """
    q (cp.ndarray): N x L x L array representing the anyon field, ∈ ℤ/2ℤ.
    x_error (cp.ndarray): N x L x L array representing the horizontal errors, ∈ ℤ/2ℤ.
    y_error (cp.ndarray): N x L x L array representing the vertical errors, ∈ ℤ/2ℤ.
    Φ (cp.ndarray): N x L x L array representing the field, ∈ ℝ.
    """

    def __init__(self, N: int, L: int):
        self.q = cp.zeros((N, L, L), dtype = cp.uint8)
        self.x_error = cp.zeros((N, L, L), dtype = cp.uint8)
        self.y_error = cp.zeros((N, L, L), dtype = cp.uint8)
        self.Φ = cp.zeros((N, L, L), dtype = cp.float32)

    def __iter__(self):
        return iter((self.q, self.x_error, self.y_error, self.Φ))
    
    @property
    def N(self) -> int:
        return self.q.shape[0]

    @property
    def L(self) -> int:
        return self.q.shape[1]
    
    @property
    def ρ(self) -> float:
        return cp.count_nonzero(self.q) / self.L ** 2 / self.N
    
    def draw(self) -> Axes:
        return plot_state(self)
    
    def add_errors(self, p_error: float) -> None:
        """
        Adds errors to the current state.

        Parameters:
        p_error (float): The probability of an X error occuring per spin.

        Returns:
        None
        """

        k = cp.array([0,1,1], dtype = cp.uint8)

        x_errors = (cp.random.random((self.N, self.L, self.L)) < p_error).astype(cp.uint8)
        horiz_anyons = correlate1d(x_errors, k, axis = 2, mode = 'wrap') % 2

        y_errors = (cp.random.random((self.N, self.L, self.L)) < p_error).astype(cp.uint8)
        vert_anyons = correlate1d(y_errors, k, axis = 1, mode = 'wrap') % 2
        
        self.q ^= vert_anyons ^ horiz_anyons
        self.x_error ^= x_errors
        self.y_error ^= y_errors
    
    def update_anyon(self) -> None:
        """
        Update the anyon state by moving to the neighbor with largest field value with probability 1/2.

        Returns:
        None
        """
        
        swv = sliding_window_view(cp.pad(self.Φ, ((0,0),(1,1),(1,1)), mode = 'wrap'), (1, 3, 3))
        direction = swv.reshape(self.N, self.L, self.L, 9)[:,:,:,1::2].argmax(axis = 3).astype(cp.uint8) # 0, 1, 2, 3
        direction += 1 # 1, 2, 3, 4 to allow for masking
        direction *= ((self.q == 1) & (cp.random.random((self.N, self.L, self.L)) < 0.5))

        left = (direction == 1) # -x
        down = (direction == 2) # -y
        up = (direction == 3) # +y
        right = (direction == 4) # +x

        self.q ^= (left | down | up | right)

        # self.q ^= cp.roll(left, -1, axis = 1)
        self.q[:,:-1,:] ^= left[:,1:,:]
        self.q[:,-1,:] ^= left[:,0,:]
        self.y_error ^= left

        # self.q ^= cp.roll(down, -1, axis = 2)
        self.q[:,:,:-1] ^= down[:,:,1:]
        self.q[:,:,-1] ^= down[:,:,0]
        self.x_error ^= down

        # self.q ^= cp.roll(up, 1, axis = 2)
        self.q[:,:,1:] ^= up[:,:,:-1]
        self.q[:,:,0] ^= up[:,:,-1]
        # self.x_error ^= cp.roll(up, 1, axis = 2)
        self.x_error[:,:,1:] ^= up[:,:,:-1]
        self.x_error[:,:,0] ^= up[:,:,-1]

        # self.q ^= cp.roll(right, 1, axis = 1)
        self.q[:,1:,:] ^= right[:,:-1,:]
        self.q[:,0,:] ^= right[:,-1,:]
        # self.y_error ^= cp.roll(right, 1, axis = 1)
        self.y_error[:,1:,:] ^= right[:,:-1,:]
        self.y_error[:,0,:] ^= right[:,-1,:]

    def update_field(self, η: float) -> None:
        """
        Update the field state.
        
        Parameters:
        η (float): Smoothing parameter, ∈ (0, 1/2).
        
        Returns:
        None
        """

        self.Φ += η / 4 * correlate(self.Φ, OONO_PURI, mode = 'wrap')
        self.Φ += self.q

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

        if p_error == 0 and state.ρ == 0:
            break

def decoder_2D_density(state: State, T: int, c: int, η: float, p_error: float) -> np.ndarray:
    """
    Run a 2D decoder on a state for T epochs.

    Parameters:
    state (State): The state to decode.
    T (int): Number of epochs to run.
    c (int): Field velocity.
    η (float): Smoothing parameter.
    p_error (float): Probability of an X error occuring per spin, per time step.

    Returns:
    np.ndarray: Array of length T containing the average density of anyons at each time step.
    """

    density = cp.empty(T, dtype = cp.float32)

    for i in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        for _ in range(c):
            state.update_field(η)
        state.update_anyon()

        density[i] = state.ρ

        if p_error == 0 and state.ρ == 0:
            break
    return density.get()

def decoder_2D_history(state: State, T: int, c: int, η: float, p_error: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a 2D decoder on a state for T epochs, keeping track of the anyon and error history.

    Parameters:
    state (State): The state to decode.
    T (int): Number of epochs to run.
    c (int): Field velocity.
    η (float): Smoothing parameter.
    p_error (float): Probability of an X error occuring per spin, per time step.

    Returns:
    np.ndarray: T x N x L x L array representing the anyon position history, ∈ ℤ/2ℤ.
    np.ndarray: T x N x L x L array representing the horizontal error history, ∈ ℤ/2ℤ.
    np.ndarray: T x N x L x L array representing the vertical error history, ∈ ℤ/2ℤ.
    """

    q_history = cp.empty((2*T, state.N, state.L, state.L), dtype = cp.uint8)
    x_error_history = cp.empty((2*T, state.N, state.L, state.L), dtype = cp.uint8)
    y_error_history = cp.empty((2*T, state.N, state.L, state.L), dtype = cp.uint8)
    for i in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        q_history[2*i,:,:,:] = state.q
        x_error_history[2*i,:,:,:] = state.x_error
        y_error_history[2*i,:,:,:] = state.y_error

        for _ in range(c):
            state.update_field(η)
        state.update_anyon()

        q_history[2*i+1,:,:,:] = state.q
        x_error_history[2*i+1,:,:,:] = state.x_error
        y_error_history[2*i+1,:,:,:] = state.y_error

        if p_error == 0 and state.ρ == 0:
            break
    return q_history.get(), x_error_history.get(), y_error_history.get()

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

def mwpm(matching: Matching, q: cp.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the minimum-weight perfect-matching correction for a given anyon state.

    Parameters:
    matching (Matching): The matching object corresponding to the toric lattice.
    q (cp.ndarray): N x L x L array representing the anyon field, ∈ ℤ/2ℤ.

    Returns:
    np.ndarray: N x L x L array representing the horizontal correction, ∈ ℤ/2ℤ.
    np.ndarray: N x L x L array representing the vertical correction, ∈ ℤ/2ℤ.
    """

    q_a = q.get() # Move array to CPU
    
    N = q_a.shape[0]
    L = q_a.shape[1]

    x_correction = np.empty((N, L, L), dtype = np.uint8)
    y_correction = np.empty((N, L, L), dtype = np.uint8)

    for n in range(N):
        correction = matching.decode(q_a[n,:,:].ravel())
        x_correction[n,:,:] = correction[:L**2].reshape(L,L)
        y_correction[n,:,:] = correction[L**2:].reshape(L,L)

    return x_correction, y_correction

def logical_error(x_error: np.ndarray, y_error: np.ndarray) -> np.ndarray:
    """
    Checks if the error configuration corresponds to a logical error.

    Parameters:
    x_error (np.ndarray): N x L x L array representing the horizontal error configuration, ∈ ℤ/2ℤ.
    y_error (np.ndarray): N x L x L array representing the vertical error configuration, ∈ ℤ/2ℤ.

    Returns:
    np.ndarray: Array of length N containing whether there is a logical error for each state, ∈ ℤ/2ℤ.
    """

    x_parity = (x_error.sum(axis = 1) % 2).astype(np.uint8)
    y_parity = (y_error.sum(axis = 2) % 2).astype(np.uint8)

    return np.count_nonzero(x_parity | y_parity, axis = 1) > 0