import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def QvaluesADvs00(t, r, p, s, epsilon, d):
    return np.array([
        [-(2*p - epsilon*p + epsilon*t) / (2 * (1 - d)), -1 / (2 * (1 - d)) * (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t)],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (1 - d)), -1 / (2 * (1 - d)) * (2*d*p - d*epsilon*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + d*epsilon*t)]
    ])

def QvaluesADvs01(t, r, p, s, epsilon, d):
    return np.array([
        [-1 / (2 * (1 - d)) * (2*p - 2*d*p - epsilon*p + 2*d*epsilon*p + 2*d*t + epsilon*t - 2*d*epsilon*t),
         -(1 / (2 * (1 - d))) * (2*d*epsilon*p - d**2*epsilon*p - d*epsilon**2*p + d**2*epsilon**2*p + epsilon*r - d*epsilon*r + 2*s - 2*d*s - epsilon*s + d*epsilon*s + 2*d*t - 2*d*epsilon*t + d**2*epsilon*t + d*epsilon**2*t - d**2*epsilon**2*t)],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (1 - d)),
         -(1 / (2 * (1 - d))) * (2*d*p - 2*d**2*p - 2*d*epsilon*p + 3*d**2*epsilon*p + d*epsilon**2*p - d**2*epsilon**2*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d**2*t + 2*d*epsilon*t - 3*d**2*epsilon*t - d*epsilon**2*t + d**2*epsilon**2*t)]
    ])

def QvaluesADvs11(t, r, p, s, epsilon, d):
    return np.array([
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (1 - d)), -1 / (2 * (1 - d)) * (d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t)],
        [-(epsilon*p + 2*t - epsilon*t) / (2 * (1 - d)), -1 / (2 * (1 - d)) * (d*epsilon*p + 2*r - 2*d*r - epsilon*r + d*epsilon*r + epsilon*s - d*epsilon*s + 2*d*t - d*epsilon*t)]
    ])

def QvaluesGTvsGT(t, r, p, s, epsilon, d):
    return np.array([
        [-1 / (2 * (1 - d)) * (2*d*p + epsilon*p - 2*d*epsilon*p + 2*t - 2*d*t - epsilon*t + 2*d*epsilon*t),
         - (2*d*epsilon*p - d*epsilon**2*p + 4*r - 4*d*r - 2*epsilon*r + 2*d*epsilon*r + 2*epsilon*s - 2*d*epsilon*s + d*epsilon**2*t) / (2 * (1 - d) * (2 - 2*d + d*epsilon))],
        [-(2*p - epsilon*p + epsilon*t) / (2 * (1 - d)),
         - (1 / (2 * (1 - d) * (2 - 2*d + d*epsilon))) * (4*d*p - 4*d**2*p - 4*d*epsilon*p + 6*d**2*epsilon*p + d*epsilon**2*p - 2*d**2*epsilon**2*p + 2*epsilon*r - 2*d*epsilon*r + 4*s - 8*d*s + 4*d**2*s - 2*epsilon*s + 6*d*epsilon*s - 4*d**2*epsilon*s + 2*d*epsilon*t - 2*d**2*epsilon*t - d*epsilon**2*t + 2*d**2*epsilon**2*t)]
    ])

def plot_q_values():
    t, s, r, p, epsilon = 1, 0, 0.9, 0.2, 0.2
    d_values = np.linspace(0, 1, 100)
    
    # Q-values for Playing C in State 0
    q00 = [QvaluesADvs00(t, r, p, s, epsilon, d)[0, 0] for d in d_values]
    q01 = [QvaluesADvs01(t, r, p, s, epsilon, d)[0, 0] for d in d_values]
    q11 = [QvaluesADvs11(t, r, p, s, epsilon, d)[0, 0] for d in d_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, q00, label='QvaluesADvs00')
    plt.plot(d_values, q01, label='QvaluesADvs01')
    plt.plot(d_values, q11, label='QvaluesADvs11')
    plt.xlabel('d')
    plt.ylabel('Q-value')
    plt.title('Q-values for Playing C in State 0')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Q-values for Playing D in State 1
    q00_d1 = [QvaluesADvs00(t, r, p, s, epsilon, d)[1, 0] for d in d_values]
    q01_d1 = [QvaluesADvs01(t, r, p, s, epsilon, d)[1, 0] for d in d_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, q00_d1, label='QvaluesADvs00 (D, State 1)')
    plt.plot(d_values, q01_d1, label='QvaluesADvs01 (D, State 1)')
    plt.xlabel('d')
    plt.ylabel('Q-value')
    plt.title('Q-values for Playing D in State 1')
    plt.legend()
    plt.grid()
    plt.show()
    
    # Q-values for Playing C in State 1
    q00_c1 = [QvaluesADvs00(t, r, p, s, epsilon, d)[1, 1] for d in d_values]
    q01_c1 = [QvaluesADvs01(t, r, p, s, epsilon, d)[1, 1] for d in d_values]
    q11_c1 = [QvaluesADvs11(t, r, p, s, epsilon, d)[1, 1] for d in d_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(d_values, q00_c1, label='QvaluesADvs00 (C, State 1)')
    plt.plot(d_values, q01_c1, label='QvaluesADvs01 (C, State 1)')
    plt.plot(d_values, q11_c1, label='QvaluesADvs11 (C, State 1)')
    plt.xlabel('d')
    plt.ylabel('Q-value')
    plt.title('Q-values for Playing C in State 1')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ =='__main__':
    
    # Example parameters (you can modify these based on your setup)
    t = 1
    r = 0.8
    p = 0.2
    s = 0
    epsilon = 0.2

    plot_q_values()