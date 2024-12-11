import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import qmc

# is the value iteration
# dynamics between the value iteration properly, look at the image

# Define Strategy Classes
class GrimTrigger_Simple:
    """Grim Trigger strategy: Cooperate until defected upon, then defect forever."""
    def __init__(self):
        self.triggered_defection = False

    def get_action(self, opponent_defected):
        if opponent_defected:
            self.triggered_defection = True
        return np.array([0, 1]) if self.triggered_defection else np.array([1, 0])
    

class epsilon_GrimTrigger:
    """Grim Trigger strategy: Cooperate unless the other player has a better Q-value for defecting."""
    def __init__(self):
        pass

    def get_action(self, epsilon_value):
        """Determine action based on Q-values."""
        # Cooperate if Q_cooperate > Q_defect, otherwise defect
        return np.array([[1-epsilon_value/2, epsilon_value/2], [epsilon_value/2, 1-epsilon_value/2]])
    
class Theoretical_QValueCalculator:
    def __init__(self, epsilon, P, R, T, S,delta):
        """
        Initialize the QValueCalculator with game parameters.
        
        Parameters:
            epsilon (float): The noise parameter (probability of random defection/cooperation).
            P (float): The payoff for both players defecting (Punishment).
            R (float): The payoff for both players cooperating (Reward).
            T (float): The payoff for defecting when the opponent cooperates (Temptation).
            S (float): The payoff for cooperating when the opponent defects (Sucker's payoff).
        """
        self.epsilon = epsilon
        self.P = P
        self.R = R
        self.T = T
        self.S = S
        self.delta = delta
    
    def calculate_c1_c2(self):
        # Calculate c1 and c2
        epsilon_value = self.epsilon
        P, R, T, S = self.P, self.R, self.T, self.S

        c2 = ((2*P) - (epsilon_value *P) - (epsilon_value *R) - (2 *S) + (epsilon_value *S) + (epsilon_value *T))/((-1 + epsilon_value) *((-2 *P) + (2 *epsilon_value *P) + (2 *S) - (epsilon_value *S) - (epsilon_value *T)))
        c1 = (-epsilon_value *P + 2 *R - epsilon_value *R + epsilon_value *S - 2 *T + epsilon_value *T)/((-1 + epsilon_value)* (-2 *P + 2 *epsilon_value *P - epsilon_value *S + 2 *T - epsilon_value *T))

        return c1, c2
    def calculate_q_values_eGT(self):
        Q11 = ((1 - self.epsilon / 2) * self.P + (self.epsilon / 2) * self.T) / (1 - self.delta)
        numerator_Q00 = (1 - self.epsilon / 2) * self.R + (self.epsilon / 2) * self.S + (self.epsilon / 2) * self.delta * Q11
        denominator_Q00 = 1 - (1 - self.epsilon / 2) * self.delta
        Q00 = numerator_Q00 / denominator_Q00
        Q10 = (1 - self.epsilon / 2) * self.T + (self.epsilon / 2) * self.P + self.delta * Q11
        Q01 = (1 - self.epsilon / 2) * self.S + (1 - self.epsilon / 2) * self.delta * Q11 + (self.epsilon / 2) * self.R + (self.epsilon / 2) * self.delta * Q00

        return Q00, Q01, Q10, Q11 # the order should remain this always but pay attention that q01 means cooperating in state 1


    def calculate_q_values_eAD(self):
        Q11 = ((1 - self.epsilon / 2) * self.P + (self.epsilon / 2) * self.T) / (1 - self.delta)
        Q10 = (1 - self.epsilon / 2)*self.T + (self.epsilon / 2) * self.P + self.delta * Q11
        Q01 = (1 - self.epsilon / 2)*self.S + (1 - self.epsilon / 2)*self.delta * Q11 + (self.epsilon / 2) * self.R + (self.epsilon / 2) * self.delta * Q10
        Q00 = ((1 - self.epsilon / 2)*self.R) + ((1 - self.epsilon / 2)*self.delta*Q10) + ((self.epsilon / 2) * self.S) + ((self.epsilon / 2) * self.delta * Q11)

        return Q00, Q01, Q10, Q11 # the order should remain this always but pay attention that q01 means cooperating in state 1

    def calculate_q_values_eAC(self):
        Q00 = 0  # initial guess for Q00
        Q01 = 0  # initial guess for Q10
        tol=1e-6
        # Iterative approach to update Q00 and Q10 until convergence
        for _ in range(100):
            Q01_new = ((1 - self.epsilon / 2) * self.S + (self.epsilon / 2) * self.R + (self.epsilon / 2) * self.delta * Q00) / (1 - (1 - self.epsilon / 2) * self.delta)

            Q00_new = ((1 - self.epsilon / 2) * self.R + (self.epsilon / 2) * self.S + (self.epsilon / 2) * self.delta * Q01_new) / (1 - (1 - self.epsilon / 2) * self.delta)

            # Check for convergence
            if abs(Q01_new - Q01) < tol and abs(Q00_new - Q00) < tol:
                break
            
            # Update Q00 and Q10 for the next iteration
            Q00 = Q00_new
            Q01 = Q01_new

        Q11 = (1 - self.epsilon/2)*self.P + (self.epsilon/2 * self.T) + self.delta * Q01
        Q10 = ((1 - self.epsilon / 2) * self.T + (self.epsilon / 2) * self.P) + self.delta * Q01
        return Q00, Q01, Q10, Q11 # the order should remain this always but pay attention that q01 means cooperating in state 1

    def calculate_q_values(self, c1, c2):
        """
        Calculate the Q-values based on the range of delta.
        
        Parameters:
            delta (float): The discount factor (future rewards weight).
            c1 (float): Lower bound for eGT calculation.
            c2 (float): Upper bound for eGT calculation.
        
        Returns:
            tuple: Q-values (Q00, Q01, Q10, Q11).
        """
        if self.delta < c1:
            print('Strategy is e-Always Defecting')
            return self.calculate_q_values_eAD()
        elif c1 <= self.delta <= c2:
            print('Strategy is e-GrimTrigger')
            return self.calculate_q_values_eGT()
        else:
            print('Strategy is e-Always Cooperating')
            return self.calculate_q_values_eAC()
        
    def print_strategy_chosen(self, c1, c2):
        if self.delta < c1:
            return 1  # e-Always Defecting
        elif c1 <= self.delta <= c2:
            return 2  # e-GrimTrigger
        else:
            return 3  # e-Always Cooperating


      
class TitForTat:
    """Tit-for-Tat strategy: Cooperate initially, then copy opponent's last move."""
    def __init__(self):
        self.previous_opponent_move = np.array([1, 0])  # Start by cooperating as a 1D array

    def get_action(self, opponent_move=None):
        if opponent_move is not None:
            self.previous_opponent_move = opponent_move  # Store as 1D array
        return self.previous_opponent_move

def convert_dict_to_mean_df(data_dict, mean = True):
    dfs = []
    for key in data_dict.keys():
        qvalues = data_dict[key].copy()
        flattened_qvalues = [array.flatten() for array in qvalues]
        stacked_qvalues = np.vstack(flattened_qvalues)
        df = pd.DataFrame(stacked_qvalues, columns=["Q1", "Q2", "Q3", "Q4"])
        dfs.append(df)
    

    if mean:
        max_length = max(len(df) for df in dfs)

        mean_rows = []

        for i in range(max_length):
            row_means = []

            for df in dfs:
                if i < len(df):  # If the row exists in this DataFrame
                    row_means.append(df.iloc[i][['Q1', 'Q2', 'Q3', 'Q4']].values)
                else:  # If the row does not exist, append NaNs
                    row_means.append([np.nan, np.nan, np.nan, np.nan])

            # Compute the mean for each column (Q1, Q2, Q3, Q4) across all DataFrames
            mean_row = np.nanmean(row_means, axis=0)  # np.nanmean ignores NaN values

            mean_rows.append(mean_row)

        # Convert the list of row-wise means into a DataFrame
        mean_df = pd.DataFrame(mean_rows, columns=['Q1', 'Q2', 'Q3', 'Q4'])
        return mean_df
    
    else:
        return dfs


def generate_unique_2x2():
    # Generate a 2x2 array with random values
    array = np.random.rand(2, 2)
    
    # Ensure the values are distinct when rounded to two decimal points
    while len(np.unique(array.round(1))) < 4:
        array = np.random.rand(2, 2)
    
    return array

def generate_unique_complete_samplings(T, delta):
    """
    Generate a 2x2 array with random values in the range [0, T / (1 - delta)]
    ensuring that all values are distinct when rounded to one decimal place.
    
    Parameters:
        T (float): Upper limit factor.
        delta (float): Factor to adjust the range.
        
    Returns:
        np.ndarray: A 2x2 array with unique rounded values.
    """
    # Calculate the upper limit
    upper_limit = T / (1 - delta)
    
    # Generate a 2x2 array with random values in the range [0, upper_limit]
    array = np.random.uniform(0, upper_limit, (2, 2))
    
    # Ensure the values are distinct when rounded to one decimal place
    while len(np.unique(array.round(1))) < 4:
        array = np.random.uniform(0, upper_limit, (2, 2))
    
    return array

def latin_hypercube_sampling(T, delta):
    
    upper_limit = T / (1 - delta)
    sampler = qmc.LatinHypercube(d=1)
    sample = sampler.random(n=4)
    unique_2x2_array = sample.reshape(2, 2)
    while len(np.unique(unique_2x2_array)) < 4:
        sample = sampler.random(n=1)
        unique_2x2_array = sample.reshape(2, 2)
    
    return unique_2x2_array

def create_line(value, length=10):
    """
    Create a line of repeated values.

    Parameters:
        value (float): The value to repeat.
        length (int): The number of times to repeat the value.

    Returns:
        list: A list containing the repeated values.
    """
    return [value for _ in range(length)]


class Game_Model_Structure:
    def __init__(self, S=2, A=2, DC=1, CC=0.8, DD=0.2, CD=0):
        """
        Initialize the game model with custom parameters.

        Parameters:
        - S (int): Number of states
        - A (int): Number of actions
        - DC (float): Reward for (D, C)
        - CC (float): Reward for (C, C)
        - DD (float): Reward for (D, D)
        - CD (float): Reward for (C, D)
        """
        self.S = S
        self.A = A
        self.DC = DC
        self.CC = CC
        self.DD = DD
        self.CD = CD
        
        # Create the reward matrix
        self.r1 = np.array([[self.CC, self.CD], [self.DC, self.DD]])
        
        # Define the probability matrices
        self.P_CC = np.array([[1, 0], [1, 0]])
        self.P_CD = np.array([[0, 1], [0, 1]])
        self.P_DC = self.P_CD  # Same as P_CD
        self.P_DD = self.P_CD  # Same as P_CD

    def get_probability_matrices(self):
        """
        Returns the probability matrices P_CC, P_CD, P_DC, P_DD.
        """
        return self.P_CC, self.P_CD, self.P_DC, self.P_DD

    def get_reward_matrix(self):
        """
        Returns the reward matrix.
        """
        return self.r1

class PrisonersDilemmaSolver:
    def __init__(self, strategy, e_initial, lambda_val, alpha, delta, K, M, e_decay, random_uniform=True, error_stop='any', verbose=False, plotting = True, check_on_value_iteration = False, gt_learned_check = False):
        self.strategy = strategy
        self.e = e_initial
        self.lambda_val = lambda_val
        self.alpha = alpha
        self.delta = delta
        self.K = K
        self.M = M
        self.e_decay = e_decay
        self.random_uniform = random_uniform
        self.error_stop = error_stop
        self.verbose = verbose
        self.plotting = plotting
        self.check_on_value_iteration = check_on_value_iteration
        self.gt_learned_check = gt_learned_check
        self.S, self.A = 2, 2
        game_structure = Game_Model_Structure(self.S, self.A, 1, 0.8, 0.2, 0)
        self.P_CC, self.P_CD, self.P_DC, self.P_DD = game_structure.get_probability_matrices()
        self.r1 = game_structure.get_reward_matrix()

    def run(self):
        q1_history_per_sample = {}
        actions_taken = []
        cooperations, defections = 0, 0
        grim_trigger_count_dict = {}
        e = self.e
        initial_q_value = []
        for m in range(self.M):
            q1_history = []
            grim_trigger_count = 0

            if self.random_uniform:
                q1 = latin_hypercube_sampling(1, self.delta)
            else:
                q1 = generate_unique_complete_samplings(1, self.delta)

            # Store initial values
            initial_q_value.append(q1.copy())
            q1_val = q1.copy()
            error = 1
            count = 0
            triggered_defection = False  # Used for Grim Trigger
            previous_action = np.array([1, 0])  # Assume initial cooperation by Player 1 for TitForTat
            
            
            while error:
                if isinstance(self.strategy, GrimTrigger_Simple):
                    pi2 = np.array([self.strategy.get_action(triggered_defection), 
                                    self.strategy.get_action(triggered_defection)]).T
                    
                elif isinstance(self.strategy, epsilon_GrimTrigger):
                    pi2 = self.strategy.get_action(epsilon_value=e).T

                elif isinstance(self.strategy, TitForTat):
                    pi2 = np.array([self.strategy.get_action(previous_action),
                                    self.strategy.get_action(previous_action)]).T
                else:
                    raise ValueError("Unsupported strategy")

                # Ensure pi2 is 2x2 for matrix multiplication

                r1_pi2 = np.dot(self.r1, pi2.T)
                P1_C_pi2 = np.array([self.P_CC[0, :] * pi2[0, 0] + self.P_CD[0,:] * pi2[1, 0], 
                                    self.P_CC[1,:] * pi2[0, 1] + self.P_CD[1,:] * pi2[1, 1]])
                P1_D_pi2 = np.array([self.P_DC[0,:] * pi2[0, 0] + self.P_DD[0,:] * pi2[1, 0], 
                                    self.P_DC[1,:] * pi2[0, 1] + self.P_DD[1,:] * pi2[1, 1]])

                for k in range(self.K):
                    
                    #q1_max = (1 - e) * np.max(q1_val, axis=0) + e * np.mean(q1_val, axis=0)
                    q1_max = np.max(q1_val, axis=0)
                    q1_new = np.zeros((2, 2))
                    q1_new[0, 0] = np.dot(P1_C_pi2[0], q1_max)
                    q1_new[0, 1] = np.dot(P1_C_pi2[1], q1_max)
                    q1_new[1, 0] = np.dot(P1_D_pi2[0], q1_max)
                    q1_new[1, 1] = np.dot(P1_D_pi2[1], q1_max)

                    # Final update to q1_new
                    q1_new = r1_pi2 + self.delta * q1_new.copy()
                    q1_val = q1_new.copy()

                
                    # Debugging logs
                    #print(f"Iteration {k}:")
                    #print(f"  r1_pi2 (rewards): {r1_pi2}")
                    #print(f"  q1_max (future value estimates): {q1_max}")
                    #print(f"  q1_new (updated Q-values): {q1_new}")

                if self.error_stop == 'any':
                    error = np.any(np.abs(q1_val - q1) > 1e-5)
                elif self.error_stop == 'mean':
                    error = np.mean(np.abs(q1_val - q1)) > 1e-5
                e *= self.e_decay


                # Update Q-values and exploration rate
                q1 += (np.random.rand() > self.lambda_val) * self.alpha * (q1_val - q1)
                q1_history.append(q1.copy()) 
                if self.check_grim_trigger(q1):
                    grim_trigger_count += 1
                action = np.argmax(q1[:, 0])  # Choose action in state 0
                actions_taken.append(action)
                
                if action == 0:
                    cooperations += 1
                else:
                    defections += 1

                # Set triggered_defection for GrimTrigger and update previous_action for TitForTat
                if action == 1:
                    triggered_defection = True  # Grim Trigger condition
                previous_action = np.array([1, 0]) if action == 0 else np.array([0, 1])  # For TitForTat

                q1_val = q1.copy()
                count += 1
            
                if self.gt_learned_check:
                    grim_trigger_count_dict[m] = grim_trigger_count
            
            q1_history_per_sample[m] = q1_history.copy()



            if self.verbose:
                print("Final Q-values for Player 1:")
                print(q1)
                if np.argmax(q1[:, 0]) == 1:
                    print("Final Q-values for Player 1 tell us that the player is Defecting")
                else: 
                    print("Final Q-values for Player 1 tell us that the player is Collaborating")


                print(f"Number of collaborations: {cooperations}, Number of defections: {defections}")
                print("Expected behavior:", "collaborate more" if self.delta >= 0.9 else "defect more")

        # Plotting results
        if self.plotting: 
            self.plot_q_values(q1_history_per_sample)

        return cooperations, defections, q1_history_per_sample, convert_dict_to_mean_df(q1_history_per_sample, mean=True), grim_trigger_count_dict, initial_q_value

    def check_value_iteration(self):
        qval_inner_value_iteration = {}
        e = self.e
        for m in range(self.M):
            q_list = []
            if self.random_uniform:
                q1 = latin_hypercube_sampling(1, self.delta)
            else:
                q1 = generate_unique_complete_samplings(1, self.delta)
            
            q1_val = q1.copy()
                
            if isinstance(self.strategy, epsilon_GrimTrigger):
                pi2 = self.strategy.get_action(epsilon_value=e) #if you transpose or not it remains identical because it is a symmetric matrix

            else:
                raise ValueError("Unsupported strategy")

            r1_pi2 = np.dot(self.r1, pi2)
            P1_C_pi2 = np.array([self.P_CC[0] * pi2[0, 0] + self.P_CD[0] * pi2[1, 0], 
                                self.P_CC[1] * pi2[0, 1] + self.P_CD[1] * pi2[1, 1]])
            P1_D_pi2 = np.array([self.P_DC[0] * pi2[0, 0] + self.P_DD[0] * pi2[1, 0], 
                                self.P_DC[1] * pi2[0, 1] + self.P_DD[1] * pi2[1, 1]])

            for k in range(self.K):
                #q1_max = (1 - e) * np.max(q1_val, axis=0) + e * np.mean(q1_val, axis=0)
                q1_max = np.max(q1_val, axis=0)
                q1_new = np.zeros((2, 2))
                q1_new[0, 0] = np.dot(P1_C_pi2[0], q1_max)
                q1_new[0, 1] = np.dot(P1_C_pi2[1], q1_max)
                q1_new[1, 0] = np.dot(P1_D_pi2[0], q1_max)
                q1_new[1, 1] = np.dot(P1_D_pi2[1], q1_max)

                # Final update to q1_new
                q1_new = r1_pi2 + self.delta * q1_new.copy()
                #q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
                #q1_new_1 = r1_pi2 + self.delta * q1_new.copy()
                q1_val = q1_new.copy()
                q_list.append(q1_val)
            qval_inner_value_iteration[m]= q_list


        return qval_inner_value_iteration
    
    def plot_values(self, ax, dfs, mean_df, theoretical_value, column, linestyle_mean, linestyle_theoretical, title, xlabel, ylabel, starting_point_print):
        """
        Plot Q-values for a specific subplot.

        Parameters:
            ax: Subplot axis to plot on.
            dfs: List of DataFrames containing individual runs.
            mean_df: DataFrame containing mean values across runs.
            theoretical_value: Theoretical Q-value to plot.
            column: The column in the DataFrames to plot.
            linestyle_mean: Line style for the mean plot.
            linestyle_theoretical: Line style for the theoretical plot.
            title: Title of the subplot.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            starting_point_print: Whether to print starting values.
        """
        for i, df in enumerate(dfs):
            if starting_point_print:
                print(f"Starting {column} for sampling {i} is {df[column][0]}")
            ax.plot(df[column], linestyle='--', label=f"{column} - Iteration {i+1}")
        ax.plot(mean_df[column], linestyle=linestyle_mean, label=f"{column} mean")
        upper_limit = 1 / (1 - self.delta) #change: insert T as input

        #ax.set_ylim([0, upper_limit])
        ax.plot(create_line(theoretical_value, len(mean_df)), linestyle=linestyle_theoretical, label=f"Theoretical {column}")

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
    
    def print_starting_conditions(dfs):
        """
        Print the starting conditions for all sampling numbers.

        Parameters:
            dfs: List of DataFrames containing individual runs.
        """
        for i, df in enumerate(dfs):
            if df['Q1'][0] > df['Q2'][0]:
                print(f'Starting Condition for sampling n° {i} is cooperating in State 0')
            elif df['Q1'][0] < df['Q2'][0]:
                print(f'Starting Condition for sampling n° {i} is defecting in State 0')
            else:
                print(f'Starting Condition for sampling n° {i} is exactly the same value for Q00 and Q01 in State 0')

            if df['Q3'][0] > df['Q4'][0]:
                print(f'Starting Condition for sampling n° {i} is cooperating in State 1')
            elif df['Q3'][0] < df['Q4'][0]:
                print(f'Starting Condition for sampling n° {i} is defecting in State 1')
            else:
                print(f'Starting Condition for sampling n° {i} is exactly the same value for Q10 and Q11 in State 1')
    
    def plot_value_iteration(self, Q00, Q01, Q10, Q11, starting_point_print=False):
        """
        Plot value iteration results for the Prisoner's Dilemma.

        Parameters:
            Q00, Q01, Q10, Q11: Theoretical Q-values for each state-action pair.
            starting_point_print: Whether to print starting values of Q-values.
        """
        qval_inner_value_iteration = self.check_value_iteration()
        dfs = convert_dict_to_mean_df(qval_inner_value_iteration, mean=False)
        mean_df = convert_dict_to_mean_df(qval_inner_value_iteration, mean=True)

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        self.plot_values(axs[0, 0], dfs, mean_df, Q00, 'Q1', ':', 'solid',
                    "Q1: Action 0, State 0: Q-values across different initializations",
                    "Iteration", "Q-value", starting_point_print)

        self.plot_values(axs[0, 1], dfs, mean_df, Q01, 'Q2', ':', 'solid',
                    "Q2: Action 1, State 0: Q-values across different initializations",
                    "Iteration", "Q-value", starting_point_print)

        self.plot_values(axs[1, 0], dfs, mean_df, Q10, 'Q3', ':', 'solid',
                    "Q3: Action 0, State 1: Q-values across different initializations",
                    "Iteration", "Q-value", starting_point_print)

        self.plot_values(axs[1, 1], dfs, mean_df, Q11, 'Q4', ':', 'solid',
                    "Q4: Action 1, State 1: Q-values across different initializations",
                    "Iteration", "Q-value", starting_point_print)

        if starting_point_print:
            self.print_starting_conditions(dfs)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_q_values(q1_history_per_sample):
        q1_history = convert_dict_to_mean_df(q1_history_per_sample, mean=True)

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].plot(q1_history['Q1'], linestyle='--', label="Q-value Action 0 (Cooperate, State 0)")
        axs[0, 0].set_title("State 0, Action 0: Q-values over Time")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Q-value")
        axs[0, 0].legend()

        axs[0, 1].plot(q1_history['Q2'], linestyle='-.', label="Q-value Action 1 (Defect, State 0)")
        axs[0, 1].set_title("State 0, Action 1: Q-values over Time")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Q-value")
        axs[0, 1].legend()

        axs[1, 0].plot(q1_history['Q3'], linestyle='--', label="Q-value Action 0 (Cooperate, State 1)")
        axs[1, 0].set_title("State 1, Action 0: Q-values over Time")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Q-value")
        axs[1, 0].legend()

        axs[1, 1].plot(q1_history['Q4'], linestyle='-.', label="Q-value Action 1 (Defect, State 1)")
        axs[1, 1].set_title("State 1, Action 1: Q-values over Time")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Q-value")
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def check_grim_trigger(q_values, tolerance=1e-5):
        """Check if Q-values match Grim Trigger behavior."""
        Q00, Q01 = q_values[0, 0], q_values[1, 0]  # State 0 (cooperate, defect)
        Q10, Q11 = q_values[0, 1], q_values[1, 1]  # State 1 (cooperate, defect)

        # Grim Trigger conditions
        condition_1 = Q00 > Q01 + tolerance  # Prefer cooperation in state 0
        condition_2 = Q10 + tolerance < Q11  # Prefer defection in state 1

        return condition_1 and condition_2
