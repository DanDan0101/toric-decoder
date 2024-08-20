import numpy as np

from scipy.sparse import csc_matrix
from pymatching import Matching

from numba import uint8, float32
from numba import jit
from numba.experimental import jitclass

from utils import roll_2d, laplace_2d

spec = [
    ('q', uint8[:,:]),
    ('x_error', uint8[:,:]),
    ('y_error', uint8[:,:]),
    ('Φ', float32[:,:]),
]
@jitclass(spec)
class State:
    """
    q (np.ndarray): L x L array representing the anyon field, ∈ ℤ/2ℤ.
    x_error (np.ndarray): L x L array representing the horizontal errors, ∈ ℤ/2ℤ.
    y_error (np.ndarray): L x L array representing the vertical errors, ∈ ℤ/2ℤ.
    Φ (np.ndarray): L x L array representing the field, ∈ ℝ.
    """

    def __init__(self, L: int):
        self.q = np.zeros((L, L), dtype = np.uint8)
        self.x_error = np.zeros((L, L), dtype = np.uint8)
        self.y_error = np.zeros((L, L), dtype = np.uint8)
        self.Φ = np.zeros((L, L), dtype = np.float32)
    
    @property
    def L(self) -> int:
        return self.q.shape[0]
    
    @property
    def ρ(self) -> float:
        return np.count_nonzero(self.q) / self.L ** 2
    
    @property
    def neighborhood(self) -> np.ndarray:
        return np.array([-self.L, -1, 1, self.L], dtype = np.int32)
    
    def add_errors(self, p_error: float) -> None:
        """
        Adds errors to the current state.

        Parameters:
        p_error (float): The probability of an X error occuring per spin.

        Returns:
        None
        """

        x_errors = (np.random.random((self.L, self.L)) < p_error).astype(np.uint8)
        horiz_anyons = x_errors ^ roll_2d(x_errors, (0, -1))

        y_errors = (np.random.random((self.L, self.L)) < p_error).astype(np.uint8)
        vert_anyons = y_errors ^ roll_2d(y_errors, (-1, 0))
        
        self.q ^= vert_anyons ^ horiz_anyons
        self.x_error ^= x_errors
        self.y_error ^= y_errors

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
            idx = self.neighborhood + base
            idx %= self.L ** 2
            neighborhood = self.Φ.take(idx)
            direction = neighborhood.argmax()
            if np.random.random() < 0.5:
                self.q[x, y] ^= 1
                if direction == 0: # -x
                    self.q[(x - 1) % self.L, y] ^= 1
                    self.y_error[x, y] ^= 1
                elif direction == 1: # -y
                    self.q[x, (y - 1) % self.L] ^= 1
                    self.x_error[x, y] ^= 1
                elif direction == 2: # +y
                    self.q[x, (y + 1) % self.L] ^= 1
                    self.x_error[x, (y + 1) % self.L] ^= 1
                elif direction == 3: # +x
                    self.q[(x + 1) % self.L, y] ^= 1
                    self.y_error[(x + 1) % self.L, y] ^= 1
                else:
                    raise ValueError("Invalid direction")

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

        if p_error == 0 and state.ρ == 0:
            break

@jit
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
    np.ndarray: Array of length T containing the density of anyons at each time step.
    """

    density = np.empty(T, dtype = np.float32)

    for i in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        for _ in range(c):
            state.update_field(η)
        state.update_anyon()

        density[i] = state.ρ

        if p_error == 0 and state.ρ == 0:
            break
    return density

@jit
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
    np.ndarray: T x L x L array representing the anyon position history, ∈ ℤ/2ℤ.
    np.ndarray: T x L x L array representing the horizontal error history, ∈ ℤ/2ℤ.
    np.ndarray: T x L x L array representing the vertical error history, ∈ ℤ/2ℤ.
    """

    q_history = np.empty((2*T, state.L, state.L), dtype = np.uint8)
    x_error_history = np.empty((2*T, state.L, state.L), dtype = np.uint8)
    y_error_history = np.empty((2*T, state.L, state.L), dtype = np.uint8)
    for i in range(T):
        if p_error > 0:
            state.add_errors(p_error)
        q_history[2*i,:,:] = state.q
        x_error_history[2*i,:,:] = state.x_error
        y_error_history[2*i,:,:] = state.y_error

        for _ in range(c):
            state.update_field(η)
        state.update_anyon()

        q_history[2*i+1,:,:] = state.q
        x_error_history[2*i+1,:,:] = state.x_error
        y_error_history[2*i+1,:,:] = state.y_error

        if p_error == 0 and state.ρ == 0:
            break
    return np.array(q_history), np.array(x_error_history), np.array(y_error_history)

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

def mwpm(matching: Matching, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the minimum-weight perfect-matching correction for a given anyon state.

    Parameters:
    matching (Matching): The matching object corresponding to the toric lattice.
    q (np.ndarray): L x L array representing the anyon field, ∈ ℤ/2ℤ.

    Returns:
    np.ndarray: L x L array representing the horizontal correction, ∈ ℤ/2ℤ.
    np.ndarray: L x L array representing the vertical correction, ∈ ℤ/2ℤ.
    """

    L = q.shape[0]
    correction = matching.decode(q.flatten())
    x_correction = correction[:L**2].reshape(L,L)
    y_correction = correction[L**2:].reshape(L,L)
    return x_correction, y_correction

@jit
def logical_error(x_error: np.ndarray, y_error: np.ndarray) -> bool:
    """
    Checks if the error configuration corresponds to a logical error.

    Parameters:
    x_error (np.ndarray): L x L array representing the horizontal error configuration, ∈ ℤ/2ℤ.
    y_error (np.ndarray): L x L array representing the vertical error configuration, ∈ ℤ/2ℤ.

    Returns:
    bool: Whether there is a logical error.
    """

    x_parity = x_error.sum(axis = 0) % 2
    y_parity = y_error.sum(axis = 1) % 2

    return x_parity.any() or y_parity.any()