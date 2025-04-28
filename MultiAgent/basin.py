import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def QvaluesADvs00(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*p - epsilon*p + epsilon*t) / (2 * (-1 + d)), -1 / (2 * (-1 + d)) * (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t)],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (-1+ d)), -1 / (2 * (-1 + d)) * (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t)]
    ])

def QvaluesADvs01(t, r, p, s, epsilon, d):
    return np.array([
        [-1 / (2 * (-1+d)) * (2*p - 2*d*p - epsilon*p + 2*d*epsilon*p + 2*d*t + epsilon*t - 2*d*epsilon*t),
         -(1 / (2 * (-1+d))) * (2*d*epsilon*p - d**2*epsilon*p - d*epsilon**2*p + d**2*epsilon**2*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + 2*d*t - 2*d*epsilon*t + d**2*epsilon*t + d*epsilon**2*t - d**2*epsilon**2*t)],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1+ d)),
         -(1 / (2 * (-1+d))) * (2*d*p - 2*d**2*p - 2*d*epsilon*p + 3*d**2*epsilon*p + d*epsilon**2*p - d**2*epsilon**2*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d**2*t + 2*d*epsilon*t - 3*d**2*epsilon*t - d*epsilon**2*t + d**2*epsilon**2*t)]
    ])

def QvaluesADvs11(t, r, p, s, epsilon, d):
    return np.array([
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1+d)), -1 / (2 * (-1+d)) * (d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t)],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1+d)), -1 / (2 * (-1+d)) * (d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t)]
    ])

def QvaluesGTvsGT(t, r, p, s, epsilon, d):
    return np.array([
        [-1 / (2 * (-1+d)) * (2*d*p + epsilon*p - 2*d*epsilon*p + 2*t - 2*d*t - epsilon*t + 2*d*epsilon*t),
         - (2*d*epsilon*p - d*epsilon**2*p + 4*r - 4*d*r - 2*epsilon*r + 2*d*epsilon*r + 2*epsilon*s - 2*d*epsilon*s + d*epsilon**2*t) / (2 * (-1+d) * (2 - 2*d + d*epsilon))],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (-1+d)),
         - (1 / (2 * (-1+d) * (2 - 2*d + d*epsilon))) * (4*d*p - 4*d**2*p - 4*d*epsilon*p + 6*d**2*epsilon*p + d*epsilon**2*p - 2*d**2*epsilon**2*p + 2*epsilon*r - 2*d*epsilon*r + 4*s - 8*d*s + 4*d**2*s - 2*epsilon*s + 6*d*epsilon*s - 4*d**2*epsilon*s + 2*d*epsilon*t - 2*d**2*epsilon*t - d*epsilon**2*t + 2*d**2*epsilon**2*t)]
    ])


class VectorFieldPlotter:
    """
    Plots a vector field and optionally overlays trajectory dynamics and BoA boundary.
    
    Parameters:
    -----------
    q1 : pd.DataFrame, optional
        DataFrame containing Q-values for player 1. Required if show_trajectory=True.
    q2 : pd.DataFrame, optional
        DataFrame containing Q-values for player 2. Required if show_trajectory=True.
    t, r, p, s : float
        Game parameters (Temptation, Reward, Punishment, Sucker).
    epsilon : float
        Learning rate for Q-value computation.
    d : float
        Discount factor.
    q_func_gt : callable
        Function that returns Q-values for GT vs GT (used for delta x).
    q_func_ad : callable
        Function that returns Q-values for AD vs strategy (used for delta y).
    alpha : float, optional
        Scaling factor for vector field direction. Default is 0.1.
    bound : float, optional
        Axis bounds for the vector field plot. Default is 1.0.
    x_state : str, optional
        State to use for the x-axis trajectory ('StateC' or 'StateD'). Default is 'StateC'.
    y_state : str, optional
        State to use for the y-axis trajectory ('StateC' or 'StateD'). Default is 'StateC'.
    show_trajectory : bool, optional
        If True, displays the trajectory from q1 and q2. Default is True.
    show_basin : bool, optional
        If True, displays the BoA boundary line. Default is True.

    
    EXAMPLE USAGE:
    plotter = VectorFieldPlotter(
        q1=my_q1_df,
        q2=my_q2_df,
        t=1, r=3, p=0, s=5,
        epsilon=0.01, d=0.9,
        q_func_gt=QvaluesGTvsGT,
        q_func_ad=QvaluesADvs11,
        x_state='StateC',
        y_state='StateD',
        show_trajectory=True,
        show_basin=True
    )
    plotter.plot()

    """
    def __init__(self, q1=None, q2=None,
                 t=1, r=3, p=0, s=5, epsilon=0.01, delta=0.9,
                 q_func_gt=None, q_func_ad=None,
                 alpha=0.1, bound=1.0,
                 x_state='StateC', y_state='StateC',
                 show_trajectory=True, show_basin=True):

        self.q1 = q1
        self.q2 = q2
        self.t, self.r, self.p, self.s = t, r, p, s
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.bound = bound
        self.x_state = x_state
        self.y_state = y_state
        self.show_trajectory = show_trajectory
        self.show_basin = show_basin
        self.eps = 1e-8

        self.q_func_gt = q_func_gt
        self.q_func_ad = q_func_ad

        if self.show_trajectory:
            if self.q1 is None or self.q2 is None:
                raise ValueError("q1 and q2 must be provided if show_trajectory is True.")
            self.prepare_states()
        else:
            self.x = self.y = self.u = self.v = None

        self.compute_deltas()

    def prepare_states(self):
        """Computes StateC and StateD from Q-values and prepares trajectory data."""
        self.q1['StateC'] = self.q1['Q1'] - self.q1['Q3']
        self.q1['StateD'] = self.q1['Q2'] - self.q1['Q4']
        self.q2['StateC'] = self.q2['Q1'] - self.q2['Q3']
        self.q2['StateD'] = self.q2['Q2'] - self.q2['Q4']

        self.x = self.q1[self.x_state].values
        self.y = self.q2[self.y_state].values
        self.u = self.x[1:] - self.x[:-1]
        self.v = self.y[1:] - self.y[:-1]

    def compute_deltas(self):
        """Computes Q-value differences and BoA boundary endpoint."""
        Q_GT = self.q_func_gt(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]
        Q_AD = self.q_func_ad(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]

        self.dx = Q_GT[0] - Q_GT[1]
        self.dy = Q_AD[0] - Q_AD[1]

        inv_scale_dx = 1 / ((1 - self.delta) * abs(self.dx) + self.eps)
        inv_scale_dy = 1 / ((1 - self.delta) * abs(self.dy) + self.eps)
        self.a = min(inv_scale_dx, inv_scale_dy)
        self.end_x = -self.a * self.dx
        self.end_y = -self.a * self.dy

    def field1(self, x, y): return self.alpha * (self.dx - x), self.alpha * (self.dy - y)
    def field2(self, x, y): return self.alpha * (self.dx - x), self.alpha * (self.dx - y)
    def field3(self, x, y): return self.alpha * (self.dy - x), self.alpha * (self.dy - y)
    def field4(self, x, y): return self.alpha * (self.dy - x), self.alpha * (self.dx - y)

    def flip_field(self, f):
        """Flips a vector field across the origin while preserving direction."""
        return lambda x, y: tuple(-v for v in f(-x, -y))

    def add_vector_field(self, ax, field_func, x_range, y_range, density=15):
        """Adds a quiver vector field to a given matplotlib axis."""
        x_vals = np.linspace(*x_range, density)
        y_vals = np.linspace(*y_range, density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U, V = field_func(X, Y)
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=4)

    def plot(self):
        """Generates the vector field plot with optional trajectory and BoA boundary."""
        fig, ax = plt.subplots(figsize=(8, 8))

        if self.show_trajectory:
            ax.quiver(self.x[:-1], self.y[:-1], self.u, self.v, angles='xy', scale_units='xy', scale=1, color='blue')
            ax.scatter(self.x, self.y, color='black', s=10)

        # Vector fields by quadrant
        self.add_vector_field(ax, self.flip_field(self.field3), (-self.bound, 0), (-self.bound, 0))  # 3rd quadrant
        self.add_vector_field(ax, self.flip_field(self.field2), (0, self.bound), (0, self.bound))    # 1st quadrant
        self.add_vector_field(ax, self.field1, (0, self.bound), (-self.bound, 0))                    # 4th quadrant
        self.add_vector_field(ax, self.field4, (-self.bound, 0), (0, self.bound))                    # 2nd quadrant

        if self.show_basin:
            ax.plot([0, self.end_x], [0, self.end_y], color='red', linewidth=2, label='BoA Boundary')

        # Axes styling
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Combined Vector Field with BoA Boundary')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()
