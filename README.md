### Payoff Matrix


\begin{array}{|c|c|c|}
\hline
\textbf{Player A / Player B} & \textbf{Cooperate (C)} & \textbf{Defect (D)} \\
\hline
\textbf{Cooperate (C)} & R = 0.8 & S = 0 \\
\hline
\textbf{Defect (D)} & T = 1.0 & P = 0.2 \\
\hline
\end{array}


### Theoretical Q-Values for Grim Trigger


\begin{aligned}
Q_{11} & = \frac{P}{1 - \delta} = \frac{0.2}{1 - 0.5} = 0.4 \\
Q_{00} & = R + \delta Q_{11} = 0.8 + 0.5 \cdot 0.4 = 1.0 \\
Q_{01} & = T + \delta Q_{11} = 1.0 + 0.5 \cdot 0.4 = 1.2 \\
Q_{10} & = S + \delta Q_{11} = 0 + 0.5 \cdot 0.4 = 0.2
\end{aligned}


### Theoretical Q-Values for Grim Trigger incorporating epsilon
$
\textbf{Q\_00)} \text{ In state 0 (neither player defected previously):}
$
With probability \(1-e\), the opponent also cooperates, leading to \( R(CC) = 0.8 \).  
With probability \(e\), the opponent defects, leading to \( R(CD) = 0 \).

Thus, the expected reward for \( R(s_0, a_0) \) is:

$
R(s_0, a_0) = (1-e) \cdot 0.8 + e \cdot 0 = 0.8(1-e)
$

So the Qvalue in 00 is $Q_{00} = R(s_0, a_0) + \delta \cdot Q_{00} = \frac{0.8(1-e)}{1-\delta}$

$
\textbf{Q\_01)} \text{ In state 0 (neither player defected previously):}
$
With probability \(1-e\), the opponent cooperates, leading to \( R(DC) = 1 \).  
With probability \(e\), the opponent defects, leading to \( R(DD) = 0.2 \).

Thus, the expected reward for \( R(s_0, a_1) \) is:

$
R(s_0, a_1) = (1-e) \cdot 1 + e \cdot 0.2 = 1 - 0.8e
$
So the Qvalue in 01 is $Q_{01} = R(s_0, a_1) + \delta \cdot Q_{11} = (1 - 0.8e) + \delta \cdot Q_{11}$
$
\textbf{Q\_10)} \text{ In state 1 (at least one player defected previously):}
$
With probability \( \frac{e}{2} \), the opponent cooperates, leading to \( R(DC) = 1 \).  
With probability \( 1 - \frac{e}{2} \), the opponent defects, leading to \( R(DD) = 0.2 \).

Thus, the expected reward for \( R(s_1, a_0) \) is:

$
R(s_1, a_0) = \frac{e}{2} \cdot 1 + \left( 1 - \frac{e}{2} \right) \cdot 0.2 = 0.2 + 0.4e
$
So the Qvalue in 10 is $Q_{10} = R(s_1, a_0) + \delta \cdot Q_{10} = \frac{0.2+0.4e}{1-\delta}$
$
\textbf{Q\_11)} \text{ In state 1 (at least one player defected previously):}
$
With probability \( \frac{e}{2} \), the opponent cooperates, leading to \( R(DC) = 1 \).  
With probability \( 1 - \frac{e}{2} \), the opponent defects, leading to \( R(DD) = 0.2 \).

Thus, the expected reward for \( R(s_1, a_1) \) is:

$
R(s_1, a_1) = \frac{e}{2} \cdot 1 + \left( 1 - \frac{e}{2} \right) \cdot 0.2 = 0.2 + 0.4e
$
\\
So the Qvalue in 11 is $Q_{11} = R(s_1, a_1) + \delta \cdot Q_{11} = \frac{0.2+0.4e}{1-\delta}$


