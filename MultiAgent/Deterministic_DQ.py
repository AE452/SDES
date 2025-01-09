import numpy as np
import time
#mordor cluster --> Mathias 

def algorithm(e, lambda_val, alpha,delta, random_uniform,K,M, verbose = False):
    # Number of players, states, and actions
    N = 2
    S = 2
    A = 2

    # Payoffs
    DC = 1
    CC = 0.8
    DD = 0.2
    CD = 0

    r1 = np.array([[CC, CD], [DC, DD]])
    r2 = r1.T

    # State transitions
    P_CC = np.array([[1, 0], [1, 0]])
    P_CD = np.array([[0, 1], [0, 1]])
    P_DC = P_CD
    P_DD = P_CD
    # Equilibria
    pi_eqC = np.array([[1-e/2, e/2], [e/2, 1-e/2]])
    pi_eqD = np.array([[e/2, e/2], [1-e/2, 1-e/2]])

    

    # Vectors/lists for recording equilibrium reached
    eqC = np.zeros((1, M))
    eqD = np.zeros((1, M))

    #Start timer
    start_time = time.time()
    #M is number of samples
    for m in range(M):
        # Initialize q values
        # ===========================================

        if random_uniform: 
            # Randomly, using a uniform distribution on [0, 1]

            q1 = np.random.rand(A, S)
            q2 = np.random.rand(A, S)
        
        else:
            # Randomly, but with a specific ordering: what ordering
            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()
            # ===========================================
            q1 = np.array([[temp2, temp3], [temp1, temp4]])

            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()

            q2 = np.array([[temp1, temp3], [temp2, temp4]])

        q1_val = q1.copy()
        q2_val = q2.copy()

        count = 0
        error = 1

        #T=int(5e2)
        #for t in range(T):
        while error:
            # Update policy pi1 and pi2 based on q values
            a1_max = np.argmax(q1, axis=0)
            pi1 = np.zeros((A, S))
            pi1[a1_max[0], 0] = 1
            pi1[a1_max[1], 1] = 1
            pi1 = (1 - e) * pi1 + e * np.ones((A, S)) / A

            a2_max = np.argmax(q2, axis=0)
            pi2 = np.zeros((A, S))
            pi2[a2_max[0], 0] = 1
            pi2[a2_max[1], 1] = 1
            pi2 = (1 - e) * pi2 + e * np.ones((A, S)) / A

            # Payoff vector
            r1_pi2 = np.dot(r1, pi2)
            r2_pi1 = np.dot(r2.T, pi1)

            # Transition probabilities
            P1_C_pi2 = np.array([P_CC[0] * pi2[0,0] + P_CD[0]* pi2[1,0], P_CC[1] * pi2[0,1] + P_CD[1] * pi2[1,1]])
            P1_D_pi2 = np.array([P_DC[0] * pi2[0,0] + P_DD[0] * pi2[1,0], P_DC[1] * pi2[0,1] + P_DD[1] * pi2[1,1]])
            P2_C_pi1 = np.array([P_CC[0] * pi1[0,0] + P_CD[0] * pi1[1,0], P_CC[1] * pi1[0,1] + P_CD[1] * pi1[1,1]])
            P2_D_pi1 = np.array([P_DC[0] * pi1[0,0] + P_DD[0] * pi1[1,0], P_DC[1] * pi1[0,1] + P_DD[1] * pi1[1,1]])

            #Calculate optimal q-values given the opponent's strategy using value iteration
            
            for k in range(K):
                #check again this part
                q1_max = ((1 - e) * np.max(q1_val, axis=0) + e * np.mean(q1_val, axis=0))
                q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
                q1_new = r1_pi2 + delta * q1_new

                q2_max = ((1 - e) * np.max(q2_val, axis=0) + e * np.mean(q2_val, axis=0))
                q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
                q2_new = r2_pi1 + delta * q2_new

                q1_val = q1_new.copy()
                q2_val = q2_new.copy()
#changed 
            error = np.any(np.abs(q1_val - q1) + np.abs(q2_val - q2) > 1e-3)

            q1 += (np.random.rand() > lambda_val) * alpha * (q1_val - q1)
            q2 += (np.random.rand() > lambda_val) * alpha * (q2_val - q2)

            q1_val = q1.copy()
            q2_val = q2.copy()

            count += 1

        eqC[0, m] = np.allclose(pi1, pi_eqC) and np.allclose(pi2, pi_eqC)
        eqD[0, m] = np.allclose(pi1, pi_eqD) and np.allclose(pi2, pi_eqD)

    eq = np.vstack((eqC, eqD, eqC + eqD))
    if verbose:
        print([np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M])

        progtime = time.time() - start_time
        if progtime < 60:
            print("My program took", round(progtime, 2), "seconds to run.")
        if progtime > 60:
            print("My program took", round(progtime/60, 2), "minutes to run.")

    return [np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M]


def algorithm_new(e, lambda_val, alpha,delta, random_uniform,tolerance, M, verbose = False):
    # Number of players, states, and actions
    N = 2
    S = 2
    A = 2

    # Payoffs --> change in the future in order to see if there are values of e and delta leading us to P and R
    DC = 1
    CC = 0.8
    DD = 0.2
    CD = 0

    r1 = np.array([[CC, CD], [DC, DD]])
    r2 = r1.T

    # State transitions
    P_CC = np.array([[1, 0], [1, 0]])
    P_CD = np.array([[0, 1], [0, 1]])
    P_DC = P_CD
    P_DD = P_CD
    # Equilibria
    pi_eqC = np.array([[1-e/2, e/2], [e/2, 1-e/2]])
    pi_eqD = np.array([[e/2, e/2], [1-e/2, 1-e/2]])

    

    # Vectors/lists for recording equilibrium reached
    eqC = np.zeros((1, M))
    eqD = np.zeros((1, M))

    #Start timer
    start_time = time.time()
    #M is number of samples
    for m in range(M):
        # Initialize q values
        # ===========================================

        if random_uniform: 
            # Randomly, using a uniform distribution on [0, 1]

            q1 = np.random.rand(A, S)
            q2 = np.random.rand(A, S)
        
        else:
            # Randomly, but with a specific ordering: what ordering
            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()
            # ===========================================
            q1 = np.array([[temp2, temp3], [temp1, temp4]])

            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()

            q2 = np.array([[temp1, temp3], [temp2, temp4]])

        q1_val = q1.copy()
        q2_val = q2.copy()

        count = 0
        #tolerance = 1e-5 # Small value to check for convergence
        error = 1  # Initialize error to a large value

        while error:
            # Update policy pi1 and pi2 based on q values
            a1_max = np.argmax(q1, axis=0)
            pi1 = np.zeros((A, S))
            pi1[a1_max[0], 0] = 1
            pi1[a1_max[1], 1] = 1
            pi1 = (1 - e) * pi1 + e * np.ones((A, S)) / A

            a2_max = np.argmax(q2, axis=0)
            pi2 = np.zeros((A, S))
            pi2[a2_max[0], 0] = 1
            pi2[a2_max[1], 1] = 1
            pi2 = (1 - e) * pi2 + e * np.ones((A, S)) / A

            # Payoff vector
            r1_pi2 = np.dot(r1, pi2)
            r2_pi1 = np.dot(r2.T, pi1)

            # Transition probabilities
            P1_C_pi2 = np.array([P_CC[0] * pi2[0,0] + P_CD[0]* pi2[1,0], P_CC[1] * pi2[0,1] + P_CD[1] * pi2[1,1]])
            P1_D_pi2 = np.array([P_DC[0] * pi2[0,0] + P_DD[0] * pi2[1,0], P_DC[1] * pi2[0,1] + P_DD[1] * pi2[1,1]])
            P2_C_pi1 = np.array([P_CC[0] * pi1[0,0] + P_CD[0] * pi1[1,0], P_CC[1] * pi1[0,1] + P_CD[1] * pi1[1,1]])
            P2_D_pi1 = np.array([P_DC[0] * pi1[0,0] + P_DD[0] * pi1[1,0], P_DC[1] * pi1[0,1] + P_DD[1] * pi1[1,1]])

            #Calculate optimal q-values given the opponent's strategy using value iteration
            q_error = 1
            while q_error > tolerance:
                q1_max = ((1 - e) * np.max(q1_val, axis=0) + e * np.mean(q1_val, axis=0))
                q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
                q1_new = r1_pi2 + delta * q1_new

                q2_max = ((1 - e) * np.max(q2_val, axis=0) + e * np.mean(q2_val, axis=0))
                q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
                q2_new = r2_pi1 + delta * q2_new

                q_error = np.max(np.abs(q1_new - q1_val) + np.abs(q2_new - q2_val))
                
                # Update q-values
                q1_val = q1_new.copy()
                q2_val = q2_new.copy()
                                

            error = np.any(np.abs(q1_val - q1) + np.abs(q2_val - q2) > 1e-3) #mathematica proofs: q_values tolerance we should try larger than value iteration smaller

            q1 += (np.random.rand() > lambda_val) * alpha * (q1_val - q1)
            q2 += (np.random.rand() > lambda_val) * alpha * (q2_val - q2)

            q1_val = q1.copy()
            q2_val = q2.copy()

            count += 1

        eqC[0, m] = np.allclose(pi1, pi_eqC) and np.allclose(pi2, pi_eqC)
        eqD[0, m] = np.allclose(pi1, pi_eqD) and np.allclose(pi2, pi_eqD)

    eq = np.vstack((eqC, eqD, eqC + eqD))
    if verbose:
        print([np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M])

        progtime = time.time() - start_time
        if progtime < 60:
            print("My program took", round(progtime, 2), "seconds to run.")
        if progtime > 60:
            print("My program took", round(progtime/60, 2), "minutes to run.")

    return [np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M]

def algorithm_max(e, lambda_val, alpha,delta, random_uniform,tolerance, M, verbose = False):
    # Number of players, states, and actions
    N = 2
    S = 2
    A = 2

    # Payoffs --> change in the future in order to see if there are values of e and delta leading us to P and R
    DC = 1
    CC = 0.8
    DD = 0.2
    CD = 0

    r1 = np.array([[CC, CD], [DC, DD]])
    r2 = r1.T

    # State transitions
    P_CC = np.array([[1, 0], [1, 0]])
    P_CD = np.array([[0, 1], [0, 1]])
    P_DC = P_CD
    P_DD = P_CD
    # Equilibria
    pi_eqC = np.array([[1-e/2, e/2], [e/2, 1-e/2]])
    pi_eqD = np.array([[e/2, e/2], [1-e/2, 1-e/2]])

    

    # Vectors/lists for recording equilibrium reached
    eqC = np.zeros((1, M))
    eqD = np.zeros((1, M))

    #Start timer
    start_time = time.time()
    #M is number of samples
    for m in range(M):
        # Initialize q values
        # ===========================================

        if random_uniform: 
            # Randomly, using a uniform distribution on [0, 1]

            q1 = np.random.rand(A, S)
            q2 = np.random.rand(A, S)
        
        else:
            # Randomly, but with a specific ordering: what ordering
            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()
            # ===========================================
            q1 = np.array([[temp2, temp3], [temp1, temp4]])

            temp1 = np.random.rand()
            temp2 = temp1 + np.random.rand()
            temp3 = np.random.rand()
            temp4 = temp3 + np.random.rand()

            q2 = np.array([[temp1, temp3], [temp2, temp4]])

        q1_val = q1.copy()
        q2_val = q2.copy()

        count = 0
        #tolerance = 1e-5 # Small value to check for convergence
        error = 1  # Initialize error to a large value

        while error:
            # Update policy pi1 and pi2 based on q values
            a1_max = np.argmax(q1, axis=0)
            pi1 = np.zeros((A, S))
            pi1[a1_max[0], 0] = 1
            pi1[a1_max[1], 1] = 1
            pi1 = (1 - e) * pi1 + e * np.ones((A, S)) / A

            a2_max = np.argmax(q2, axis=0)
            pi2 = np.zeros((A, S))
            pi2[a2_max[0], 0] = 1
            pi2[a2_max[1], 1] = 1
            pi2 = (1 - e) * pi2 + e * np.ones((A, S)) / A

            # Payoff vector
            r1_pi2 = np.dot(r1, pi2)
            r2_pi1 = np.dot(r2.T, pi1)

            # Transition probabilities
            P1_C_pi2 = np.array([P_CC[0] * pi2[0,0] + P_CD[0]* pi2[1,0], P_CC[1] * pi2[0,1] + P_CD[1] * pi2[1,1]])
            P1_D_pi2 = np.array([P_DC[0] * pi2[0,0] + P_DD[0] * pi2[1,0], P_DC[1] * pi2[0,1] + P_DD[1] * pi2[1,1]])
            P2_C_pi1 = np.array([P_CC[0] * pi1[0,0] + P_CD[0] * pi1[1,0], P_CC[1] * pi1[0,1] + P_CD[1] * pi1[1,1]])
            P2_D_pi1 = np.array([P_DC[0] * pi1[0,0] + P_DD[0] * pi1[1,0], P_DC[1] * pi1[0,1] + P_DD[1] * pi1[1,1]])

            #Calculate optimal q-values given the opponent's strategy using value iteration
            q_error = 1
            while q_error > tolerance :
                q1_max = np.max(q1_val, axis=0)
                q1_new = np.vstack((np.dot(P1_C_pi2, q1_max), np.dot(P1_D_pi2, q1_max)))
                q1_new = r1_pi2 + delta * q1_new

                q2_max = np.max(q2_val, axis=0)
                q2_new = np.vstack((np.dot(P2_C_pi1, q2_max), np.dot(P2_D_pi1, q2_max)))
                q2_new = r2_pi1 + delta * q2_new

                q_error = np.max(np.abs(q1_new - q1_val)+ np.abs(q2_new - q2_val))
                
                # Update q-values
                q1_val = q1_new.copy()
                q2_val = q2_new.copy()
                                

            error = np.any(np.abs(q1_val - q1) + np.abs(q2_val - q2) > 1e-3) #mathematica proofs: q_values tolerance we should try larger than value iteration smaller

            q1 += (np.random.rand() > lambda_val) * alpha * (q1_val - q1)
            q2 += (np.random.rand() > lambda_val) * alpha * (q2_val - q2)

            q1_val = q1.copy()
            q2_val = q2.copy()

            count += 1

        eqC[0, m] = np.allclose(pi1, pi_eqC) and np.allclose(pi2, pi_eqC)
        eqD[0, m] = np.allclose(pi1, pi_eqD) and np.allclose(pi2, pi_eqD)

    eq = np.vstack((eqC, eqD, eqC + eqD))
    if verbose:
        print([np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M])

        progtime = time.time() - start_time
        if progtime < 60:
            print("My program took", round(progtime, 2), "seconds to run.")
        if progtime > 60:
            print("My program took", round(progtime/60, 2), "minutes to run.")

    return [np.sum(eqC) / M, np.sum(eqD) / M, (np.sum(eqC) + np.sum(eqD))/ M]
