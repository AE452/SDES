import numpy as np
import time
import sys
sys.path.append('../')
from utils import *

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


class Algorithm:
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
        timeout = 60 
        for m in range(self.M):
            q1, q2 = self.initialize_q_values()
            q1_val = q1.copy()
            q2_val = q2.copy()
            q1_history, q2_history = [], []
            error = 1
            loop_start_time = time.time()
            batch_start_time = time.time()

            # Outer loop with timeout
            while error:
                if time.time() - loop_start_time > 1 * timeout:  
                    # Save results even if timeout occurs
                    q1_history_per_sample[m] = q1_history.copy()
                    q2_history_per_sample[m] = q2_history.copy()

                    #print("Stopping outer loop due to timeout.")
                    break  # Exit outer loop if timeout is exceeded

                pi1, pi2 = self.update_policies(q1, q2)
                q_error = 1
                q_loop_start_time = time.time()

                # Inner loop with timeout
                while q_error > self.tolerance:
                    if time.time() - q_loop_start_time > 1 * timeout:  
                        #print("Stopping inner loop due to timeout.")
                        q1_history.append(q1.copy())
                        q2_history.append(q2.copy())
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
                if time.time() - q_loop_start_time > 1*60:
                    continue

            # Store history for each sample
            q1_history_per_sample[m] = q1_history.copy()
            q2_history_per_sample[m] = q2_history.copy()
            batch_end_time = time.time()  # End time for this batch
            batch_times[m] = batch_end_time - batch_start_time

            # Check for equilibrium conditions
            self.eqC[0, m] = np.allclose(pi1, self.state_transition.pi_eqC) and np.allclose(pi2, self.state_transition.pi_eqC)
            self.eqD[0, m] = np.allclose(pi1, self.state_transition.pi_eqD) and np.allclose(pi2, self.state_transition.pi_eqD)

            # Print summary if verbose is enabled
            if verbose:
                print(self.summary(start_time))

            # If the outer loop timed out, continue to the next sample
            if time.time() - loop_start_time > 1 * 60:
                continue

        return q1_history_per_sample, q2_history_per_sample, batch_times


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
        timeout = 60 
        for m in range(self.M):
            q1, q2 = self.initialize_q_values()
            q1_val = q1.copy()
            q2_val = q2.copy()
            q1_history, q2_history = [], []
            error = 1
            loop_start_time = time.time()
            batch_start_time = time.time()

            # Outer loop with timeout
            while error:
                if time.time() - loop_start_time > 1 * timeout:  
                    # Save results even if timeout occurs
                    q1_history_per_sample[m] = q1_history.copy()
                    q2_history_per_sample[m] = q2_history.copy()

                    #print("Stopping outer loop due to timeout.")
                    break  # Exit outer loop if timeout is exceeded

                pi1, pi2 = self.update_policies(q1, q2)
                q_error = 1
                q_loop_start_time = time.time()

                # Inner loop with timeout
                while q_error > self.tolerance:
                    if time.time() - q_loop_start_time > 1 * timeout:  
                        #print("Stopping inner loop due to timeout.")
                        q1_history.append(q1.copy())
                        q2_history.append(q2.copy())
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
                if time.time() - q_loop_start_time > 1 * 60:
                    continue

            # Store history for each sample
            q1_history_per_sample[m] = q1_history.copy()
            q2_history_per_sample[m] = q2_history.copy()
            batch_end_time = time.time()  # End time for this batch
            batch_times[m] = batch_end_time - batch_start_time

            # Check for equilibrium conditions
            self.eqC[0, m] = np.allclose(pi1, self.state_transition.pi_eqC) and np.allclose(pi2, self.state_transition.pi_eqC)
            self.eqD[0, m] = np.allclose(pi1, self.state_transition.pi_eqD) and np.allclose(pi2, self.state_transition.pi_eqD)

            # Print summary if verbose is enabled
            if verbose:
                print(self.summary(start_time))

            # If the outer loop timed out, continue to the next sample
            if time.time() - loop_start_time > 1 * 60:
                continue

        return q1_history_per_sample, q2_history_per_sample, batch_times


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
        q1_new = (1-self.alpha)*r1_pi2 + self.delta * q1_new

        q2_max = ((1 - self.e) * np.max(q2_val, axis=0) + self.e * np.mean(q2_val, axis=0))
        q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
        q2_new = (1-self.alpha)*r2_pi1 + self.delta * q2_new

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
