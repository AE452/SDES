import numpy as np
import time
import sys
sys.path.append('../')
from utils import *
from basin import *
# ============================================================
# What changed compared to the previous version?
#
# Original implementation (generate_q_for_strategy):
# - Initialized Q-values randomly.
# - Enforced inequalities for the chosen strategy
#   (e.g., AD: Q1 < Q3 and Q2 < Q4).
# - This ensured the Q-matrix matched the *strategic definition*,
#   BUT it did NOT ensure that ΔQ = (Q_defect − Q_coop) signs
#   matched the expected vector field quadrants.
# - Result: trajectories sometimes started in the wrong quadrant.
#
# New implementation (added steps):
# 1. get_signs(...)
#    - Computes the expected signs (x_sign, y_sign) of ΔQ
#      for each state directly from the theoretical model.
#
# 2. enforce_q_sign_for_state(...)
#    - Checks the actual ΔQ sign from the initialized Q-matrix.
#    - If inconsistent with the expected sign, swaps the entries
#      so the sign matches.
#    - Guarantees player incentives (defect vs cooperate)
#      are aligned with the vector fields.
#
# 3. initialize_q_values(...) updated:
#    - Before: only generated Q with inequalities.
#    - Now:
#        (a) generate Q with inequalities,
#        (b) compute expected signs via get_signs,
#        (c) enforce them with enforce_q_sign_for_state,
#        (d) store the corrected Q as the initial state.
#
# Logic of the change:
# - Before: only strategy inequalities were enforced.
# - After: BOTH strategy inequalities AND correct ΔQ signs
#   are enforced → initial trajectories always start in the
#   same quadrant as shown in the vector field plots.
# ============================================================
# ============================================================
# explanation about the FIXED INITIALIZATION: enforce_q_sign_for_state
# ============================================================
# Quadrants of ΔQ (defect − coop) in the plane (x=player1, y=player2)
#
#          ΔQ_y > 0  (player2 prefers defect)
#                   ↑
#                   |
#   QII             |            QI
# (x<0, y>0)        |        (x>0, y>0)
#  P1 → cooperate   |         both → defect
#  P2 → defect      |
#                   |
# ------------------+------------------→ ΔQ_x > 0 (player1 prefers defect)
#                   |                            
#                   |
#  QIII             |            QIV
# (x<0, y<0)        |        (x>0, y<0)
# both → cooperate  |   P1 → defect
#                   |   P2 → cooperate
#
# Interpretation:
# - ΔQ_x > 0 → player1 prefers defect in the current state
# - ΔQ_x < 0 → player1 prefers cooperate
# - ΔQ_y > 0 → player2 prefers defect
# - ΔQ_y < 0 → player2 prefers cooperate
#
# Usage:
# - get_signs(...) → returns (x_sign, y_sign) expected from the model
# - enforce_q_sign_for_state(...) → forces initial Q to match those signs
#   → ensures trajectories start in the correct quadrant shown in the vector fields
# ============================================================

func_map = {
    "AD": QvaluesADvsAD,
    "GT": QvaluesGTvsGT,
    "AGT": QvaluesADvsAGT,
    "AC": QvaluesADvsAC
}
def enforce_q_sign_for_state(q, desired_sign, state_idx):
    """
    Enforce ΔQ sign (defect - coop) in one state by swapping
    the two action entries in that column if necessary.
    q has shape (2,2) with q[action, state].
    """
    cur = q[1, state_idx] - q[0, state_idx]
    if desired_sign == "pos" and cur < 0:
        q[0, state_idx], q[1, state_idx] = q[1, state_idx], q[0, state_idx]
    elif desired_sign == "neg" and cur > 0:
        q[0, state_idx], q[1, state_idx] = q[1, state_idx], q[0, state_idx]
    return q


def get_signs(s1, s2, t, r, p, s, epsilon, delta, state_idx):
    """
    Calcola i segni teorici (ΔQ = defect - coop) per player1 (x) e player2 (y)
    nello stato state_idx, usando le funzioni Qvalues.
    """
    func1 = func_map[s2]  # player1 reagisce a strategy2
    func2 = func_map[s1]  # player2 reagisce a strategy1

    Qx = func1(t, r, p, s, epsilon, delta)[state_idx]  # [defect, coop]
    Qy = func2(t, r, p, s, epsilon, delta)[state_idx]

    dx = Qx[0] - Qx[1]
    dy = Qy[0] - Qy[1]
    x_sign = "pos" if dx >= 0 else "neg"
    y_sign = "pos" if dy >= 0 else "neg"
    return x_sign, y_sign
    
class PayoffMatrix:
    """Represents the payoff matrix for the game."""
    def __init__(self, DC=1, CC=0.8, DD=0.2, CD=0):
        self.r1 = np.array([[CC, CD], [DC, DD]])
        self.r2 = self.r1.T


class StateTransition:
    """Handles state transitions based on the game dynamics."""
    def __init__(self, e):
        self.e = e
        self.P_CC = np.array([[1, 0], [1, 0]])
        self.P_CD = np.array([[0, 1], [0, 1]])
        self.P_DC = self.P_CD
        self.P_DD = self.P_CD
        self.pi_eqC = np.array([[1 - e / 2, e / 2], [e / 2, 1 - e / 2]])
        self.pi_eqD = np.array([[e / 2, e / 2], [1 - e / 2, 1 - e / 2]])

strategies = ["AD", "GT", "AGT", "AC"]

def get_signs(s1, s2, t, r, p, s, epsilon, delta, state_idx):
    """
    Calcola i segni di ΔQ (defect - coop) per entrambi i giocatori
    nello stato `state_idx` (0 = StateC, 1 = StateD).

    """
    func1 = func_map[s2]  # x-component uses opponent strategy s2
    func2 = func_map[s1] 

    Qx = func1(t, r, p, s, epsilon, delta)[state_idx]  # valori per player x
    Qy = func2(t, r, p, s, epsilon, delta)[state_idx]  # valori per player y
    
    dx = Qx[0] - Qx[1]  # defect − coop
    dy = Qy[0] - Qy[1]

    x_sign = 'pos' if dx >= 0 else 'neg'
    y_sign = 'pos' if dy >= 0 else 'neg'
    return x_sign, y_sign


def generate_q_for_strategy(strategy, delta, T=1.0, x_sign=None, y_sign=None):
    """
    Generate a 2x2 Q-matrix that satisfies the inequalities
    corresponding to a given strategy, and optionally enforce
    ΔQ signs for x and y so that trajectories start in the
    correct quadrant.
    
    Parameters
    ----------
    strategy : str
        One of {"AD","GT","AGT","AC"}.
    delta : float
        Discount factor.
    T : float
        Scaling parameter (default 1.0).
    x_sign, y_sign : {"pos","neg",None}
        Desired signs of ΔQ = (defect − coop) for x (columns) and y (rows).
        If None, no correction is applied.
    """
    upper_limit = T / (1 - delta)
    
    # Start with random values
    Q = np.random.uniform(0, upper_limit, (2, 2))
    # Q = [ [Q00, Q01],[Q10, Q11] ]
    # Qij = Q[action i, state j]

    # Enforce inequalities depending on strategy
    #Q00 (cooperate) and Q10 (defect) is state BC
    #Q01 (cooperate) and Q11 (defect) is state A1D
    if strategy == "AD":   # Q00 < Q10 and Q01 < Q11
        if not (Q[0,0] < Q[1,0]): Q[0,0], Q[1,0] = sorted([Q[0,0], Q[1,0]])
        if not (Q[0,1] < Q[1,1]): Q[0,1], Q[1,1] = sorted([Q[0,1], Q[1,1]])

    elif strategy == "GT": # Q00 > Q10 and Q01 < Q11
        if not (Q[0,0] > Q[1,0]): Q[0,0], Q[1,0] = sorted([Q[0,0], Q[1,0]], reverse=True)
        if not (Q[0,1] < Q[1,1]): Q[0,1], Q[1,1] = sorted([Q[0,1], Q[1,1]])

    elif strategy == "AGT": # Q00 > Q10 and Q01 < Q11
        if not (Q[0,0] < Q[1,0]): Q[0,0], Q[1,0] = sorted([Q[0,0], Q[1,0]])
        if not (Q[0,1] > Q[1,1]): Q[0,1], Q[1,1] = sorted([Q[0,1], Q[1,1]], reverse=True)

    elif strategy == "AC":  # Q00 > Q10 and Q01 > Q11
        if not (Q[0,0] > Q[1,0]): Q[0,0], Q[1,0] = sorted([Q[0,0], Q[1,0]], reverse=True)
        if not (Q[0,1] > Q[1,1]): Q[0,1], Q[1,1] = sorted([Q[0,1], Q[1,1]], reverse=True)


    return Q

#TODO: create function here that is checking if the ordering remains the same till a certain point and in case plot till that point, based on the above funcion,
# flag to plot the entire history no matter the ordering, or just till the ordering is the same, useful for vector fields
#  


class Algorithm:
    """Encapsulates the algorithm for calculating equilibria."""
    def __init__(self, e, lambda_val, alpha, delta, init_mode, tolerance, M, strategy1= None,  strategy2= None, verbose=False):
        self.e = e
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.delta = delta
        self.init_mode = init_mode
        if self.init_mode == "fixed":
            if strategy1 is None or strategy2 is None:
                raise ValueError("For init_mode='fixed', you must provide strategy1 and strategy2 lists.")
            self.strategy1 = strategy1
            self.strategy2 = strategy2
        self.tolerance = tolerance
        self.M = M
        self.verbose = verbose

        self.N = 2  # Number of players
        self.S = 2  # Number of states
        self.A = 2  # Number of actions

        self.payoff = PayoffMatrix()
        self.state_transition = StateTransition(e)

        self.eqC = np.zeros((1, M))
        self.eqD = np.zeros((1, M))


    def initialize_q_values(self):
        """
        Initializes the Q-values for the two agents.

        Parameters
        ----------
        mode : str
            "random" (default) → keep using LHS/complete sampling
            "fixed" → use one of the 8 predefined systematic cases
        case_id : int, optional
            Required if mode="fixed". An integer 0–7 specifying which configuration to use.

        Returns
        -------
        q1, q2 : list[float], list[float]
            Initial Q-values for agent 1 and agent 2.
        """

        if self.init_mode == "latin":
            
            q1 = latin_hypercube_sampling(1, self.delta)
            q2 = latin_hypercube_sampling(1, self.delta)
        elif self.init_mode == "uniform":
            q1 = generate_unique_complete_samplings(1, self.delta)
            q2 = generate_unique_complete_samplings(1, self.delta)

        elif self.init_mode == "fixed":
            # Ensure the user passed the strategies
            if not hasattr(self, "strategy1") or not hasattr(self, "strategy2"):
                raise ValueError("For init_mode='fixed', you must provide strategy1 and strategy2 as strings.")
            if not isinstance(self.strategy1, str) or not isinstance(self.strategy2, str):
                raise TypeError("strategy1 and strategy2 must be strings, e.g., 'AD', 'GT'")

            


            # generate candidate q1,q2
            q1 = generate_q_for_strategy(self.strategy1, self.delta)
            q2 = generate_q_for_strategy(self.strategy2, self.delta)

            # enforce ΔQ signs for both states
            #for state_idx in (0, 1):
                # segni desiderati dal modello teorico
            #    x_sign, y_sign = get_signs(
            #        self.strategy1, self.strategy2, 1, 0.8, 0.2, 0, self.e, self.delta, state_idx
            #    )

                # correzione su q1 (player1, asse x)
            #    q1 = enforce_q_sign_for_state(q1, x_sign, state_idx)
                # correzione su q2 (player2, asse y)
            #    q2 = enforce_q_sign_for_state(q2, y_sign, state_idx)

        else:
            raise ValueError(f"Unknown mode: {self.init_mode}, must be 'latin' or 'uniform' or 'fixed'")

        return q1, q2

    def run(self, verbose=False):
        """Runs the main algorithm."""
        start_time = time.time()
        q1_history_per_sample, q2_history_per_sample, batch_times = {}, {}, {}
        timeout_out_flag_dict, timeout_vi_flag_dict = {}, {} 
        timeout_outer = 60 
        timeout_inner = 30
        
        for m in range(self.M):
            timeout_out_flag = False  # Outer loop timeout flag
            timeout_vi_flag = False  # Inner loop timeout flag
            q1, q2 = self.initialize_q_values()
            q1_val = q1.copy()
            q2_val = q2.copy()
            q1_history, q2_history = [], []
            error = 1
            loop_start_time = time.time()
            batch_start_time = time.time()

            # Outer loop with timeout
            while error:
                if time.time() - loop_start_time > 1 * timeout_outer:  
                    # Save results even if timeout occurs
                    q1_history_per_sample[m] = q1_history.copy()
                    q2_history_per_sample[m] = q2_history.copy()
                    timeout_out_flag = True
                    timeout_out_flag_dict[m] = timeout_out_flag

                    #print("Stopping outer loop due to timeout.")
                    break  # Exit outer loop if timeout is exceeded

                pi1, pi2 = self.update_policies(q1, q2)
                q_error = 1
                q_loop_start_time = time.time()

                # Inner loop with timeout
                while q_error > self.tolerance:
                    if time.time() - q_loop_start_time > 1 * timeout_inner:  
                        #print("Stopping inner loop due to timeout.")
                        q1_history.append(q1.copy())
                        q2_history.append(q2.copy())
                        timeout_vi_flag = True
                        timeout_vi_flag_dict[m] = timeout_vi_flag
                        break  # Exit inner loop if timeout is exceeded

                    # Update Q-values
                    q1_val, q2_val, q_error = self.update_q_values(pi1, pi2, q1_val, q2_val)

                # Check if the outer loop should continue based on q_error condition or timeout
                error = np.any(np.abs(q1_val - q1) + np.abs(q2_val - q2) > 1e-3)

                # Update Q-values for next iteration of outer loop
                q1 += (np.random.rand() > self.lambda_val) * self.alpha * (q1_val - q1)
                q2 += (np.random.rand() > self.lambda_val) * self.alpha * (q2_val - q2)
                q1_history.append(q1.copy())
                q2_history.append(q2.copy())

                # If inner loop timed out, continue to next outer loop iteration
                if time.time() - q_loop_start_time > 1*timeout_inner:
                    continue

            # Store history for each sample
            q1_history_per_sample[m] = q1_history.copy()
            q2_history_per_sample[m] = q2_history.copy()
            batch_end_time = time.time()  # End time for this batch
            batch_times[m] = batch_end_time - batch_start_time
            timeout_out_flag_dict[m] = timeout_out_flag
            timeout_vi_flag_dict[m] = timeout_vi_flag

            # Check for equilibrium conditions
            self.eqC[0, m] = np.allclose(pi1, self.state_transition.pi_eqC) and np.allclose(pi2, self.state_transition.pi_eqC)
            self.eqD[0, m] = np.allclose(pi1, self.state_transition.pi_eqD) and np.allclose(pi2, self.state_transition.pi_eqD)

            # Print summary if verbose is enabled
            if verbose:
                print(self.summary(start_time))

            # If the outer loop timed out, continue to the next sample
            if time.time() - loop_start_time > 1 * timeout_outer:
                continue

        return q1_history_per_sample, q2_history_per_sample, batch_times, timeout_out_flag_dict, timeout_vi_flag_dict


    def update_policies(self, q1, q2):
        """Updates the policies based on Q-values."""
        pi1 = self.calculate_policy(q1)
        pi2 = self.calculate_policy(q2)
        return pi1, pi2

    def calculate_policy(self, q):
        """Calculates the policy from Q-values."""
        a_max = np.argmax(q, axis=0)
        pi = np.zeros_like(q)
        pi[a_max[0], 0] = 1
        pi[a_max[1], 1] = 1
        return (1 - self.e) * pi + self.e * np.ones_like(pi) / self.A

    def update_q_values(self, pi1, pi2, q1_val, q2_val):
        """Updates Q-values using value iteration."""
        r1_pi2 = np.dot(self.payoff.r1, pi2)
        r2_pi1 = np.dot(self.payoff.r2.T, pi1)

        P1_C_pi2, P1_D_pi2 = self.calculate_transition_probabilities(pi2)
        P2_C_pi1, P2_D_pi1 = self.calculate_transition_probabilities(pi1)

        q1_max = ((1 - self.e) * np.max(q1_val, axis=0) + self.e * np.mean(q1_val, axis=0))
        q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
        q1_new = r1_pi2 + self.delta * q1_new

        q2_max = ((1 - self.e) * np.max(q2_val, axis=0) + self.e * np.mean(q2_val, axis=0))
        q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
        q2_new = r2_pi1 + self.delta * q2_new

        q_error = np.max(np.abs(q1_new - q1_val) + np.abs(q2_new - q2_val))
        return q1_new, q2_new, q_error

    def calculate_transition_probabilities(self, pi):
        """Calculates state transition probabilities."""
        P1_C_pi = np.array([self.state_transition.P_CC[0] * pi[0, 0] + self.state_transition.P_CD[0] * pi[1, 0],
                            self.state_transition.P_CC[1] * pi[0, 1] + self.state_transition.P_CD[1] * pi[1, 1]])
        P1_D_pi = np.array([self.state_transition.P_DC[0] * pi[0, 0] + self.state_transition.P_DD[0] * pi[1, 0],
                            self.state_transition.P_DC[1] * pi[0, 1] + self.state_transition.P_DD[1] * pi[1, 1]])
        return P1_C_pi, P1_D_pi

    def summary(self, start_time):
        """Summarizes the results and timing."""
        eq = np.vstack((self.eqC, self.eqD, self.eqC + self.eqD))
        result = [np.sum(self.eqC) / self.M, np.sum(self.eqD) / self.M, (np.sum(self.eqC) + np.sum(self.eqD)) / self.M]

        if self.verbose:
            print(result)
            progtime = time.time() - start_time
            print("My program took", round(progtime, 2), "seconds to run.")

        return result

class Algorithm_1:
    """Encapsulates the algorithm for calculating equilibria."""
    def __init__(self, e, lambda_val, alpha, delta, random_uniform, tolerance, M, verbose=False):
        self.e = e
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.delta = delta
        self.random_uniform = random_uniform
        self.tolerance = tolerance
        self.M = M
        self.verbose = verbose

        self.N = 2  # Number of players
        self.S = 2  # Number of states
        self.A = 2  # Number of actions

        self.payoff = PayoffMatrix()
        self.state_transition = StateTransition(e)

        self.eqC = np.zeros((1, M))
        self.eqD = np.zeros((1, M))

    def initialize_q_values(self):
        """Initializes the Q-values randomly."""
        if self.random_uniform:
                q1 = latin_hypercube_sampling(1, self.delta)
                q2 = latin_hypercube_sampling(1, self.delta)
        else:
            q1 = generate_unique_complete_samplings(1, self.delta)
            q2 = generate_unique_complete_samplings(1, self.delta)

        return q1, q2
    

    def run(self, verbose=False):
        """Runs the main algorithm."""
        start_time = time.time()
        q1_history_per_sample, q2_history_per_sample, batch_times = {}, {}, {}
        timeout_out_flag_dict, timeout_vi_flag_dict = {}, {} 
        timeout_outer = 60 
        timeout_inner = 30
        
        for m in range(self.M):
            timeout_out_flag = False  # Outer loop timeout flag
            timeout_vi_flag = False  # Inner loop timeout flag
            q1, q2 = self.initialize_q_values()
            q1_val = q1.copy()
            q2_val = q2.copy()
            q1_history, q2_history = [], []
            error = 1
            loop_start_time = time.time()
            batch_start_time = time.time()

            # Outer loop with timeout
            while error:
                if time.time() - loop_start_time > 1 * timeout_outer:  
                    # Save results even if timeout occurs
                    q1_history_per_sample[m] = q1_history.copy()
                    q2_history_per_sample[m] = q2_history.copy()
                    timeout_out_flag = True
                    timeout_out_flag_dict[m] = timeout_out_flag

                    #print("Stopping outer loop due to timeout.")
                    break  # Exit outer loop if timeout is exceeded

                pi1, pi2 = self.update_policies(q1, q2)
                q_error = 1
                q_loop_start_time = time.time()

                # Inner loop with timeout
                while q_error > self.tolerance:
                    if time.time() - q_loop_start_time > 1 * timeout_inner:  
                        #print("Stopping inner loop due to timeout.")
                        q1_history.append(q1.copy())
                        q2_history.append(q2.copy())
                        timeout_vi_flag = True
                        timeout_vi_flag_dict[m] = timeout_vi_flag
                        break  # Exit inner loop if timeout is exceeded

                    # Update Q-values
                    q1_val, q2_val, q_error = self.update_q_values(pi1, pi2, q1_val, q2_val)

                # Check if the outer loop should continue based on q_error condition or timeout
                error = np.any(np.abs(q1_val - q1) + np.abs(q2_val - q2) > 1e-3)

                # Update Q-values for next iteration of outer loop
                q1 += (np.random.rand() > self.lambda_val) * self.alpha * (q1_val - q1)
                q2 += (np.random.rand() > self.lambda_val) * self.alpha * (q2_val - q2)
                q1_history.append(q1.copy())
                q2_history.append(q2.copy())

                # If inner loop timed out, continue to next outer loop iteration
                if time.time() - q_loop_start_time > 1*timeout_inner:
                    continue

            # Store history for each sample
            q1_history_per_sample[m] = q1_history.copy()
            q2_history_per_sample[m] = q2_history.copy()
            batch_end_time = time.time()  # End time for this batch
            batch_times[m] = batch_end_time - batch_start_time
            timeout_out_flag_dict[m] = timeout_out_flag
            timeout_vi_flag_dict[m] = timeout_vi_flag

            # Check for equilibrium conditions
            self.eqC[0, m] = np.allclose(pi1, self.state_transition.pi_eqC) and np.allclose(pi2, self.state_transition.pi_eqC)
            self.eqD[0, m] = np.allclose(pi1, self.state_transition.pi_eqD) and np.allclose(pi2, self.state_transition.pi_eqD)

            # Print summary if verbose is enabled
            if verbose:
                print(self.summary(start_time))

            # If the outer loop timed out, continue to the next sample
            if time.time() - loop_start_time > 1 * timeout_outer:
                continue

        return q1_history_per_sample, q2_history_per_sample, batch_times, timeout_out_flag_dict, timeout_vi_flag_dict



    def update_policies(self, q1, q2):
        """Updates the policies based on Q-values."""
        pi1 = self.calculate_policy(q1)
        pi2 = self.calculate_policy(q2)
        return pi1, pi2

    def calculate_policy(self, q):
        """Calculates the policy from Q-values."""
        a_max = np.argmax(q, axis=0)
        pi = np.zeros_like(q)
        pi[a_max[0], 0] = 1
        pi[a_max[1], 1] = 1
        return (1 - self.e) * pi + self.e * np.ones_like(pi) / self.A

    def update_q_values(self, pi1, pi2, q1_val, q2_val):
        """Updates Q-values using value iteration."""
        r1_pi2 = np.dot(self.payoff.r1, pi2)
        r2_pi1 = np.dot(self.payoff.r2.T, pi1)

        P1_C_pi2, P1_D_pi2 = self.calculate_transition_probabilities(pi2)
        P2_C_pi1, P2_D_pi1 = self.calculate_transition_probabilities(pi1)

        q1_max = ((1 - self.e) * np.max(q1_val, axis=0) + self.e * np.mean(q1_val, axis=0))
        q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
        q1_new = (1-self.delta)*r1_pi2 + self.delta * q1_new

        q2_max = ((1 - self.e) * np.max(q2_val, axis=0) + self.e * np.mean(q2_val, axis=0))
        q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
        q2_new = (1-self.delta)*r2_pi1 + self.delta * q2_new

        q_error = np.max(np.abs(q1_new - q1_val) + np.abs(q2_new - q2_val))
        return q1_new, q2_new, q_error

    def calculate_transition_probabilities(self, pi):
        """Calculates state transition probabilities."""
        P1_C_pi = np.array([self.state_transition.P_CC[0] * pi[0, 0] + self.state_transition.P_CD[0] * pi[1, 0],
                            self.state_transition.P_CC[1] * pi[0, 1] + self.state_transition.P_CD[1] * pi[1, 1]])
        P1_D_pi = np.array([self.state_transition.P_DC[0] * pi[0, 0] + self.state_transition.P_DD[0] * pi[1, 0],
                            self.state_transition.P_DC[1] * pi[0, 1] + self.state_transition.P_DD[1] * pi[1, 1]])
        return P1_C_pi, P1_D_pi

    def summary(self, start_time):
        """Summarizes the results and timing."""
        eq = np.vstack((self.eqC, self.eqD, self.eqC + self.eqD))
        result = [np.sum(self.eqC) / self.M, np.sum(self.eqD) / self.M, (np.sum(self.eqC) + np.sum(self.eqD)) / self.M]

        if self.verbose:
            print(result)
            progtime = time.time() - start_time
            print("My program took", round(progtime, 2), "seconds to run.")

        return result
