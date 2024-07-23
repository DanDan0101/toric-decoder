import numpy as np
from scipy.ndimage import laplace

import seaborn as sns
rocket = sns.color_palette("rocket", as_cmap=True)

import matplotlib.pyplot as plt

BASESHIFT = np.array([-1, 0, 1])

class State:
    """
    L: lattice size
    N: number of anyons
    q: anyons, ∈ ℤ/2ℤ
    Φ: field, ∈ ℝ
    """
    def __init__(self, L: int, N: int, q: np.ndarray):
        self.L = L
        self.N = N
        self.q = q
        self.Φ = np.zeros((L, L), dtype = np.float32)
    def draw(self) -> None:
        """
        Plot the state.

        Returns:
        None
        """
        plt.matshow(self.q, cmap = rocket)
        plt.colorbar()
        plt.show()

def init_state(L: int, p_anyon: float) -> State:
    """
    Initializes a state with a random distribution of anyons.

    Parameters:
    L (int): The size of the lattice.
    p_anyon (float): The probability of an anyon being present at a given site.

    Returns:
    State: The initialized state object, with zero field.
    """

    vert_anyons = (np.random.rand(L, L) < p_anyon).astype(np.int32)
    vert_anyons += np.roll(vert_anyons, 1, axis=0)

    horiz_anyons = (np.random.rand(L, L) < p_anyon).astype(np.int32)
    horiz_anyons += np.roll(horiz_anyons, 1, axis=1)
    
    q = (vert_anyons + horiz_anyons) % 2
    N = np.sum(q)
    return State(L, N, q)

def update_field(state: State, η: float) -> None:
    """
    Update the field state.
    
    Parameters:
    state (State): The current state.
    η (float): Smoothing parameter, ∈ (0, 1/2).
    
    Returns:
    None
    """

    state.Φ += η / 4 * laplace(state.Φ, mode='wrap')
    state.Φ += state.q

def update_anyon(state: State) -> None:
    """
    Update the anyon state by moving to the neighbor with largest field value with probability 1/2.

    Parameters:
    state (State): The current state.

    Returns:
    None
    """
    
    for x, y in np.transpose(np.nonzero(state.q)):
        idx = x * state.L + y
        shifts = np.add.outer(BASESHIFT * state.L, BASESHIFT).flatten()
        neighborhood = state.Φ.take(shifts + idx, mode = 'wrap')
        neighborhood = neighborhood[1::2]
        direction = neighborhood.argmax()
        if np.random.rand() < 0.5:
            state.q[x, y] -= 1
            state.q[x, y] %= 2
            if direction == 0:
                state.q[(x - 1) % state.L, y] += 1
                state.q[(x - 1) % state.L, y] %= 2
            elif direction == 1:
                state.q[x, (y - 1) % state.L] += 1
                state.q[x, (y - 1) % state.L] %= 2
            elif direction == 2:
                state.q[x, (y + 1) % state.L] += 1
                state.q[x, (y + 1) % state.L] %= 2
            elif direction == 3:
                state.q[(x + 1) % state.L, y] += 1
                state.q[(x + 1) % state.L, y] %= 2
            else:
                raise ValueError("Invalid direction")

