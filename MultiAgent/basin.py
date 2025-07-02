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

#new functions:
def QvaluesGTvs00(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*d*p + epsilon*p - 2*d*epsilon*p + 2*t - 2*d*t - epsilon*t + 2*d*epsilon*t) / (2 * (-1 + d)),
         -(2*d*epsilon*p - d*epsilon**2*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon**2*t) / (2 * (-1 + d))],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (-1 + d)),
         -(2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t) / (2 * (-1 + d))]
    ])

def QvaluesGTvs01(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*d*p + epsilon*p - 2*d*epsilon*p + 2*t - 2*d*t - epsilon*t + 2*d*epsilon*t) / (2 * (-1 + d)),
         -(2*d*epsilon*p - d*epsilon**2*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon**2*t) / (2 * (-1 + d))],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1 + d)),
         -(d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t) / (2 * (-1 + d))]
    ])
def Qvalues01vs01(t, r, p, s, epsilon, d):
    return np.array([
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1 + d)),
         -(d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t) / (2 * (-1 + d))],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (-1 + d)),
         -(d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t) / (2 * (-1 + d))]
    ])
def Qvalues00vs00(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*p + 2*t) / (2 * (-1 + d)),
         -(2*d*p + 2*r + 2*s + 2*d*t) / (2 * (-1 + d))],
        [-(2*p + 2*t) / (2 * (-1 + d)),
         -(2*d*p + 2*r + 2*s + 2*d*t) / (2 * (-1 + d))]
    ])
def Qvalues00vsAD(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*p + 2*t) / (2 * (-1 + d)),
         -(2*d*p + 2*r + 2*s + 2*d*t) / (2 * (-1 + d))],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (-1 + d)),
         -(2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t) / (2 * (-1 + d))]
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
                 x_state='StateC', y_state='StateC',method = "quiver",cmap=cm.viridis,
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
        self.method = method
        self.cmap = cmap

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

        "TODO: the error in the trajectory and the quadrants is due to an error in this part, not in the quadrants of the vector field"
        #self.q1['StateC'] = self.q1['Q1'] - self.q1['Q3']
        #self.q1['StateD'] = self.q1['Q2'] - self.q1['Q4']
        #elf.q2['StateC'] = self.q2['Q1'] - self.q2['Q3']
        #self.q2['StateD'] = self.q2['Q2'] - self.q2['Q4']

        # Assuming Q1 = Q00, Q2 = Q10, Q3 = Q01, Q4 = Q11

# State differences (action 1 - action 0)
        self.q1['StateC'] = self.q1['Q3'] - self.q1['Q1']
        self.q1['StateD'] = self.q1['Q4'] - self.q1['Q2']

        self.q2['StateC'] = self.q2['Q3'] - self.q2['Q1']
        self.q2['StateD'] = self.q2['Q4'] - self.q2['Q2']

        # Trajectories
        self.x = self.q1[self.x_state].values  # e.g., StateC
        self.y = self.q2[self.y_state].values  # e.g., StateD

        # Displacements
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

    def field1(self, x, y):
        # Quadrant 4 (x>0, y<0)
        return self.alpha * (self.dx - x), self.alpha * (self.dy - y)

    def field2(self, x, y):
        # Quadrant 3 (x<0, y<0)
        return self.alpha * (self.dx - x), self.alpha * (self.dx - y)

    def field3(self, x, y):
        # Quadrant 1 (x>0, y>0)
        return self.alpha * (self.dy - x), self.alpha * (self.dy - y)

    def field4(self, x, y):
        # Quadrant 2 (x<0, y>0)
        return self.alpha * (self.dy - x), self.alpha * (self.dx - y)



    def flip_field(self, f):
        """Flips a vector field across the origin while preserving direction."""
        return lambda x, y: tuple(-v for v in f(-x, -y))

    def add_vector_field(self, ax, field_func, x_range, y_range, density=15, method="quiver"):
        x_vals = np.linspace(*x_range, density)
        y_vals = np.linspace(*y_range, density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U, V = field_func(X, Y)
        magnitude = np.sqrt(U**2 + V**2)
        if self.method == "quiver":
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=4)

        elif method == "streamplot":
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Flow Magnitude')

    def add_vector_field(self, ax, field_func, x_range, y_range, density=15):
        x_vals = np.linspace(x_range[0], x_range[1], density)
        y_vals = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U, V = field_func(X, Y)
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=4)

    def plot(self):
        """Generates the vector field plot with optional trajectory and BoA boundary."""
        fig, ax = plt.subplots(figsize=(8, 8))
        """
        if self.show_trajectory:
            ax.quiver(self.x[:-1], self.y[:-1], self.u, self.v, angles='xy', scale_units='xy', scale=1, color='blue')
            ax.scatter(self.x, self.y, color='black', s=10)
        """

        if self.show_trajectory:
            #ax.plot(self.x, self.y, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Trajectory')
            ax.quiver(self.x[:-1], self.y[:-1], self.u, self.v, angles='xy', scale_units='xy', scale=1, color='grey', alpha=0.5, label='Direction')
            ax.scatter(self.x[0], self.y[0], color='green', marker='o', s=40, label='Start')
            ax.scatter(self.x[-1], self.y[-1], color='red', marker='X', s=50, label='End')


        bound = self.bound

        self.add_vector_field(ax, self.field1, (0, bound), (-bound, 0))     # vector1, quadrante 4
        self.add_vector_field(ax, self.field2, (-bound, 0), (-bound, 0))    # vector2, quadrante 3
        self.add_vector_field(ax, self.field3, (0, bound), (0, bound))      # vector3, quadrante 1
        self.add_vector_field(ax, self.field4, (-bound, 0), (0, bound))     # vector4, quadrante 2




        
        if self.show_basin:
            ax.plot([0, self.end_x], [0, self.end_y], color='red', linewidth=2, label='BoA Boundary')

        # Axes styling
        """
        old part
        -----------
        -----------
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Combined Vector Field with BoA Boundary')
        ax.grid(True)
        """
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)
        ax.set_aspect('equal')
        ax.set_xlabel(f"Agent 1: Pr({self.x_state[-1]})", fontsize=12)
        ax.set_ylabel(f"Agent 2: Pr({self.y_state[-1]})", fontsize=12)
        ax.set_title('Strategy Vector Field', fontsize=14)
        # Center reference lines
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(0, color='gray', linestyle=':', alpha=0.5)

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)



        ax.legend()
        plt.tight_layout()
        plt.show()



class VectorFieldPlotter_single:
    """
    Plots a vector field derived from Q-value deltas and optionally overlays trajectory dynamics and BoA boundary.
    """
    def __init__(self, q1=None, q2=None,
                 t=1, r=3, p=0, s=5, epsilon=0.01, delta=0.9,
                 q_func_gt=None, q_func_ad=None,
                 alpha=0.1, x_state='StateC', y_state='StateC',
                 method="quiver", cmap=cm.viridis,
                 show_trajectory=True, show_basin=True):

        self.q1 = q1
        self.q2 = q2
        self.t, self.r, self.p, self.s = t, r, p, s
        self.epsilon = epsilon
        self.delta = delta
        self.alpha = alpha
        self.x_state = x_state
        self.y_state = y_state
        self.method = method
        self.cmap = cmap
        self.show_trajectory = show_trajectory
        self.show_basin = show_basin
        self.eps = 1e-8

        self.q_func_gt = q_func_gt
        self.q_func_ad = q_func_ad

        if self.show_trajectory:
            if self.q1 is None or self.q2 is None:
                raise ValueError("q1 and q2 must be provided if show_trajectory is True.")
            self.prepare_states()

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
        """Computes delta Q-values and defines the plotting range."""
        Q_GT = self.q_func_gt(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]
        Q_AD = self.q_func_ad(self.t, self.r, self.p, self.s, self.epsilon, self.delta)[0]

        self.dx = Q_GT[0] - Q_GT[1]
        self.dy = Q_AD[0] - Q_AD[1]

        self.bound = 1 / (1 - self.delta)
        self.end_x = self.dx
        self.end_y = self.dy

    def unified_field(self, x, y):
        """Single global vector field: alpha * (dx - x, dy - y)"""
        return self.alpha * (self.dx - x), self.alpha * (self.dy - y)

    def add_vector_field(self, ax, field_func, x_range, y_range, density=20):
        x_vals = np.linspace(*x_range, density)
        y_vals = np.linspace(*y_range, density)
        X, Y = np.meshgrid(x_vals, y_vals)
        U, V = field_func(X, Y)

        magnitude = np.sqrt(U**2 + V**2)
        if self.method == "quiver":
            ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=3, alpha=0.8)
        elif self.method == "streamplot":
            norm = plt.Normalize(magnitude.min(), magnitude.max())
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            ax.streamplot(X, Y, U, V, color=magnitude, cmap=self.cmap, linewidth=1)
            plt.colorbar(sm, ax=ax, label='Flow Magnitude')

    def plot(self):
        """Main plotting function for the vector field and optional overlays."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot unified vector field over symmetric domain
        self.add_vector_field(ax, self.unified_field, (-self.bound, self.bound), (-self.bound, self.bound))

        # Optional: plot trajectory
        if self.show_trajectory:
            ax.quiver(self.x[:-1], self.y[:-1], self.u, self.v,
                      angles='xy', scale_units='xy', scale=1,
                      color='gray', alpha=0.5, label='Trajectory')
            ax.scatter(self.x[0], self.y[0], color='green', marker='o', s=40, label='Start')
            ax.scatter(self.x[-1], self.y[-1], color='red', marker='X', s=50, label='End')

        # Optional: plot BoA boundary (line from (0,0) to (dx, dy))
        if self.show_basin:
            ax.plot([0, self.end_x], [0, self.end_y], color='red', linewidth=2, label='BoA Boundary')

        # Axes setup
        ax.set_xlim(-self.bound, self.bound)
        ax.set_ylim(-self.bound, self.bound)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.set_xlabel(f"Agent 1: Pr({self.x_state[-1]})", fontsize=12)
        ax.set_ylabel(f"Agent 2: Pr({self.y_state[-1]})", fontsize=12)
        ax.set_title('Strategy Vector Field', fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.show()
