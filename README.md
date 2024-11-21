### Payoff Matrix

The Payoff Matrix:

| **Player A / Player B** | **Cooperate (C)** | **Defect (D)** |
|-------------------------|-------------------|----------------|
| **Cooperate (C)**       | R = 0.8           | S = 0          |
| **Defect (D)**          | T = 1.0           | P = 0.2        |


# E-GT: Exploration in Game Theory

## States
The state space is defined as:
$$
O_0 = \{C, C\}, \quad O_1 = \{D, (C, D), (D, D)\}
$$

## Payoff Matrix
The payoff matrix for the game is:
$$
\begin{array}{c|c|c|}
    & C & D \\
\hline
C & R = 0.5 & S = 0 \\
D & T = 1 & P = -0.2 \\
\hline
\end{array}
$$

## Q-value Updates for Opponent E-GT

### For \( O_0 \):
1. Cooperation in \( O_0 \):
   $$
   Q_{00} = (1 - \frac{\varepsilon}{2}) \Big[R + \delta \max \{Q_{00}, Q_{10}\}\Big] + \frac{\varepsilon}{2} \xi
   $$

2. Defection in \( O_0 \):
   $$
   Q_{01} = (1 - \frac{\varepsilon}{2}) \Big[T + \delta \max \{Q_{01}, Q_{11}\}\Big] + \frac{\varepsilon}{2} \rho
   $$

### For \( O_1 \):
1. Cooperation in \( O_1 \):
   $$
   Q_{10} = (1 - \frac{\varepsilon}{2}) \Big[S + \delta \max \{Q_{00}, Q_{10}\}\Big] + \frac{\varepsilon}{2} \xi
   $$

2. Defection in \( O_1 \):
   $$
   Q_{11} = (1 - \frac{\varepsilon}{2}) \Big[P + \delta \max \{Q_{01}, Q_{11}\}\Big] + \frac{\varepsilon}{2} T
   $$

## Scenarios for Different Opponent Strategies

### Case 1: Opponent Plays \( CC \)
If the opponent cooperates entirely (\( CC \)), the Q-values are updated as:
$$
Q_{00} = (1 - \frac{\varepsilon}{2}) \Big[R + \delta Q_{00}\Big] + \frac{\varepsilon}{2} S
$$
$$
Q_{01} = (1 - \frac{\varepsilon}{2}) \Big[T + \delta Q_{11}\Big] + \frac{\varepsilon}{2} P
$$
$$
Q_{10} = (1 - \frac{\varepsilon}{2}) \Big[S + \delta Q_{10}\Big] + \frac{\varepsilon}{2} R
$$
$$
Q_{11} = (1 - \frac{\varepsilon}{2}) \Big[P + \delta Q_{01}\Big] + \frac{\varepsilon}{2} T
$$

### Case 2: Opponent Plays \( AD \)
For alternating cooperation and defection (\( AD \)):
$$
Q_{00} = (1 - \frac{\varepsilon}{2}) \Big[R + \delta Q_{00}\Big] + \frac{\varepsilon}{2} S
$$
$$
Q_{01} = (1 - \frac{\varepsilon}{2}) \Big[T + \delta Q_{11}\Big] + \frac{\varepsilon}{2} P
$$
$$
Q_{10} = (1 - \frac{\varepsilon}{2}) \Big[S + \delta Q_{10}\Big] + \frac{\varepsilon}{2} R
$$
$$
Q_{11} = (1 - \frac{\varepsilon}{2}) \Big[P + \delta Q_{01}\Big] + \frac{\varepsilon}{2} T
$$

### Case 3: Opponent Plays \( AC \)
If the opponent alternates cooperation (\( AC \)), the same Q-value updates apply.

### Case 4: Opponent Plays \( AG \)
If the opponent uses a general strategy (\( AG \)):
$$
Q_{00} = (1 - \frac{\varepsilon}{2}) \Big[R + \delta Q_{00}\Big] + \frac{\varepsilon}{2} \xi
$$
$$
Q_{01} = (1 - \frac{\varepsilon}{2}) \Big[T + \delta Q_{11}\Big] + \frac{\varepsilon}{2} P
$$
$$
Q_{10} = (1 - \frac{\varepsilon}{2}) \Big[S + \delta Q_{10}\Big] + \frac{\varepsilon}{2} R
$$
$$
Q_{11} = (1 - \frac{\varepsilon}{2}) \Big[P + \delta Q_{01}\Big] + \frac{\varepsilon}{2} T
$$

