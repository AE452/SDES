import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define Q-value functions (you will need to implement them from the Mathematica expressions)
def Qvalues_AD_vs_00(t, r, p, s, epsilon, d):
    # Replace with the full implementation for the Q-value functions from Mathematica
    q1 = -((2 * p - epsilon * p + epsilon * t) / (2 * (-1 + d)))
    q2 = -((2 * d * p - d * epsilon * p + epsilon * r - d * epsilon * r + 2 * s - 2 * d * s - epsilon * s + d * epsilon * s + d * epsilon * t) / (2 * (-1 + d)))
    return np.array([[q1, q2], [q1, q2]])

def Qvalues_AD_vs_01(t, r, p, s, epsilon, d):
    # Implement the Q-values for this case based on Mathematica code
    pass

def Qvalues_AD_vs_11(t, r, p, s, epsilon, d):
    # Implement the Q-values for this case based on Mathematica code
    pass

# Example parameters (you can modify these based on your setup)
t = 1
r = 0.9
p = 0.2
s = 0
epsilon = 0.2

def plot_Qvalue_difference():
    d_values = np.linspace(0, 1, 100)  # Define the range for d (inertia parameter)
    
    # Compute the Q-value differences
    Q_diff_1 = Qvalues_AD_vs_00(t, r, p, s, epsilon, d_values)[0][0] - Qvalues_AD_vs_00(t, r, p, s, epsilon, d_values)[0][1]
    Q_diff_2 = Qvalues_AD_vs_00(t, r, p, s, epsilon, d_values)[1][0] - Qvalues_AD_vs_00(t, r, p, s, epsilon, d_values)[1][1]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(d_values, Q_diff_1, label="Q-value diff (state 0, D vs C)")
    plt.plot(d_values, Q_diff_2, label="Q-value diff (state 1, D vs C)", linestyle='--')
    plt.xlabel("Inertia (d)")
    plt.ylabel("Q-value Difference")
    plt.title("Q-value Difference for AD vs 00")
    plt.legend()
    plt.grid(True)
    plt.show()

def vector_field_plot():
    # Define the grid for the vector field
    x = np.linspace(-1/(1 - 0.7), 1/(1 - 0.7), 20)  # For example, choose the range for x and y
    y = np.linspace(-1/(1 - 0.7), 1/(1 - 0.7), 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute the vector components (use the Q-value difference expressions for vectors)
    U = 0.3 * (Qvalues_GT_vs_GT(t, r, p, s, epsilon, 0.7)[0][0] - Qvalues_GT_vs_GT(t, r, p, s, epsilon, 0.7)[0][1] - X)
    V = 0.3 * (Qvalues_AD_vs_00(t, r, p, s, epsilon, 0.7)[1][0] - Qvalues_AD_vs_00(t, r, p, s, epsilon, 0.7)[1][1] - Y)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, scale=10, color=cm.viridis(np.linalg.norm([U, V], axis=0)))
    plt.xlabel("State 0 Q-value difference (x)")
    plt.ylabel("State 1 Q-value difference (y)")
    plt.title("Vector Field for Q-value Difference")
    plt.grid(True)
    plt.show()

