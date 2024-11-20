### Payoff Matrix

The Payoff Matrix:

| **Player A / Player B** | **Cooperate (C)** | **Defect (D)** |
|-------------------------|-------------------|----------------|
| **Cooperate (C)**       | R = 0.8           | S = 0          |
| **Defect (D)**          | T = 1.0           | P = 0.2        |

### Theoretical Q-Values for Grim Trigger

The Q-values are:

$$
Q_{11} = \frac{P}{1 - \delta} = \frac{0.2}{1 - 0.5} = 0.4
$$

$$
Q_{00} = R + \delta Q_{11} = 0.8 + 0.5 \cdot 0.4 = 1.0
$$

$$
Q_{01} = T + \delta Q_{11} = 1.0 + 0.5 \cdot 0.4 = 1.2
$$

$$
Q_{10} = S + \delta Q_{11} = 0 + 0.5 \cdot 0.4 = 0.2
$$

### Theoretical Q-Values for Grim Trigger incorporating epsilon

For **Q_00** in state 0 (neither player defected previously):

$$
R(s_0, a_0) = (1 - e) \cdot 0.8 + e \cdot 0 = 0.8(1 - e)
$$

Thus, the Q-value in 00 is:

$$
Q_{00} = \frac{0.8(1 - e)}{1 - \delta}
$$

For **Q_01** in state 0 (neither player defected previously):

$$
R(s_0, a_1) = (1 - e) \cdot 1 + e \cdot 0.2 = 1 - 0.8e
$$

So, the Q-value in 01 is:

$$
Q_{01} = (1 - 0.8e) + \delta \cdot Q_{11}
$$

For **Q_10** in state 1 (at least one player defected previously):

$$
R(s_1, a_0) = \frac{e}{2} \cdot 1 + \left(1 - \frac{e}{2}\right) \cdot 0.2 = 0.2 + 0.4e
$$

So, the Q-value in 10 is:

$$
Q_{10} = \frac{0.2 + 0.4e}{1 - \delta}
$$

For **Q_11** in state 1 (at least one player defected previously):

$$
R(s_1, a_1) = \frac{e}{2} \cdot 1 + \left(1 - \frac{e}{2}\right) \cdot 0.2 = 0.2 + 0.4e
$$

Thus, the Q-value in 11 is:

$$
Q_{11} = \frac{0.2 + 0.4e}{1 - \delta}
$$


OR:

### Bellman Equations for Theoretical Optimal Q-Values

The theoretical optimal Q-values for Grim Trigger with \(\epsilon\)-greedy exploration are calculated using the following Bellman equations:

                    q1_max = (1 - e) * np.max(q1_val, axis=0) + e * np.mean(q1_val, axis=0)



$$
Q_{11} &= P + \delta \left[ (1 - \epsilon) Q_{11} + \epsilon \cdot \frac{1}{4} (Q_{11} + Q_{00} + Q_{01} + Q_{10}) \right], \\
Q_{00} &= R + \delta \left[ (1 - \epsilon) Q_{00} + \epsilon \cdot \frac{1}{4} (Q_{11} + Q_{00} + Q_{01} + Q_{10}) \right], \\
Q_{01} &= T + \delta \left[ (1 - \epsilon) Q_{11} + \epsilon \cdot \frac{1}{4} (Q_{11} + Q_{00} + Q_{01} + Q_{10}) \right], \\
Q_{10} &= S + \delta \left[ (1 - \epsilon) Q_{11} + \epsilon \cdot \frac{1}{4} (Q_{11} + Q_{00} + Q_{01} + Q_{10}) \right].
$$


### Description:
- \(Q_{11}\): Q-value for mutual cooperation.
- \(Q_{00}\): Q-value for mutual defection.
- \(Q_{01}\): Q-value when the agent defects and the opponent cooperates.
- \(Q_{10}\): Q-value when the agent cooperates and the opponent defects.
- \(P, R, T, S\): Payoffs from the game matrix.
- \(\delta\): Discount factor (\(0 \leq \delta < 1\)).
- \(\epsilon\): Probability of taking a random (exploratory) action.

### Usage:
The system of equations is solved iteratively or symbolically to compute the theoretical Q-values under the Grim Trigger strategy, accounting for \(\epsilon\)-greedy exploration.
