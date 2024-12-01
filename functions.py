import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd



def animating(ptc_list, t_f, wdt, D):

    def animate(i):
        plt.cla()
        plt.hist(ptc_list[i % t_f], bins=np.linspace(0, D, int(D/5)), width=wdt, color='tab:blue')
        plt.xlabel("Displacement")
        plt.xlim([-5, D+5])

    fig = plt.figure()
    plt.hist(ptc_list[0], bins=np.linspace(0, D, int(D/5)), width=wdt, color='tab:blue')
    ani = animation.FuncAnimation(fig, animate, frames=t_f, interval=100, repeat=False)
    ani.save('results/simulation.gif')




def RW_simulation_constant_signal(n, t_f, stepsize, D, k, thresh, n_bin):
    """a simulation of random walk of a group of particles, with decay and a constant signal of π/2.
    This function returns an array composed of vectors of positions of each particle for each time frame.

    :parameter n number of particles.
    :parameter D length of the cross-section of the stem.
    :parameter t_f number of time frames.
    :parameter thresh threshold concentration.
    :parameter n_bin number of segments.
    :parameter stepsize in x-axis, usually 1.
    :parameter k number of particles to kill and create at each iteration.
    :returns: an array of positions of all particles, as a function of time."""


    # initial settings

    def p(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin((theta) - np.pi / 2)) / 2


    def theta(t, delta, theta_list):
        return delta * 0.005 + theta_list[t - 1]

    def dc(N_L, N_0):
        return ((N_L - N_0)[t - 1] + 0.0001) / ((N_L + N_0)[t - 1] + 0.0001)


    ptc_list = list(np.zeros(t_f))
    ptc_list[0] = np.random.randint(0, D, size=n, dtype=int)
    N_0, N_L = np.zeros(t_f), np.zeros(t_f)
    theta_list = np.zeros(t_f)
    delta = np.zeros(t_f)
    bins_list = np.linspace(0, D, n_bin + 1, dtype=int)    # setting up the bins list
    c_list = np.zeros((t_f, n_bin))   # setting the concentrations list
    probabilities = np.full((t_f, n), 0.5)    # setting the probabilities array
    p_list, H = np.full((t_f, n_bin), 0.5), np.zeros((t_f, n_bin))



    for t in range(1, t_f):

        # adding / removing particles. for now, k particle at a time.
        for i in range(k):    # remove
            who = np.random.randint(0, len(ptc_list[t - 1]))
            ptc_list[t - 1] = np.delete(ptc_list[t - 1], who)

        for i in range(k):    # add
            ptc_list[t - 1] = np.append(arr=ptc_list[t - 1], values=np.random.randint(0, D))



        # simulation
        curr_change = np.array([np.random.choice([-stepsize, stepsize], p=[1 - p_i, p_i]) for p_i in probabilities[t - 1]])
        ptc_list[t] = ptc_list[t - 1] + curr_change


        # calculating theta and delta
        delta[t] = dc(N_L, N_0)
        theta_list[t] = theta(t, delta[t], theta_list)   # updating theta(t) list


        # calculating concentrations and probabilities
        for j in range(n_bin):
            where = np.where(np.logical_and(ptc_list[t - 1] >= bins_list[j], ptc_list[t - 1] < bins_list[j + 1]))[0]
            c_list[t, j] = len(where) / n
            probabilities[t, where] = p(theta_list[t], thresh, c_list[t, j])


            ## new - for sanity check
            p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])
            H[t, j] = np.heaviside(c_list[t, j] - thresh, 1)



        # checking the particles at the boundaries and making sure no one passes them.
        ptc_list[t][np.where(ptc_list[t - 1] < 0)] = 0
        ptc_list[t][np.where(ptc_list[t - 1] > D)] = D

        # note number of particles at 1st and last quarter
        N_L[t] = len(np.where(ptc_list[t-1] > int(3 * D / 4))[0])
        N_0[t] = len(np.where(ptc_list[t-1] < int(D / 4))[0])


    pd.DataFrame(ptc_list).to_csv("results/ptc_list.csv")   # save the array of particles' positions
    pd.DataFrame(delta).to_csv("results/delta.csv")   # save the delta list
    pd.DataFrame(theta_list).to_csv("results/theta.csv")   # save the list of angles

    return ptc_list, delta, theta_list



def RW_simulation_changing_signal_once(n, t_f, stepsize, D, k, thresh, n_bin, tau_s):
    """a simulation of random walk of a group of particles, with decay and a signal of π/2 that lasts for tau_s seconds,
    and then changes to 0 for the rest of the simulation. This function returns an array composed of vectors of positions
     of all particle for each time frame.

    :parameter tau_s the duration of the signal.
    :parameter n number of particles.
    :parameter D length of the cross-section of the stem.
    :parameter t_f number of time frames.
    :parameter thresh threshold concentration.
    :parameter n_bin number of segments.
    :parameter stepsize in x-axis, usually 1.
    :parameter k number of particles to kill and create at each iteration.
    :returns: an array of positions of all particles, as a function of time."""


    # initial settings

    def p(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin(theta - np.pi / 2)) / 2


    def p_no_signal(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin(theta)) / 2

    def theta(t, delta, theta_list):
        return delta * 0.005 + theta_list[t - 1]


    def dc(N_L, N_0):
        return ((N_L - N_0)[t - 1] + 0.0001) / ((N_L + N_0)[t - 1] + 0.0001)


    ptc_list = list(np.zeros(t_f))
    ptc_list[0] = np.random.randint(0, D, size=n, dtype=int)
    N_0, N_L = np.zeros(t_f), np.zeros(t_f)
    theta_list = np.zeros(t_f)
    delta = np.zeros(t_f)
    bins_list = np.linspace(0, D, n_bin + 1, dtype=int)    # setting up the bins list
    c_list = np.zeros((t_f, n_bin))   # setting the concentrations list
    probabilities = np.full((t_f, n), 0.5)    # setting the probabilities array
    p_list, H = np.full((t_f, n_bin), 0.5), np.zeros((t_f, n_bin))



    for t in range(1, t_f):

        # adding / removing particles. for now, k particle at a time.
        for i in range(k):    # remove
            who = np.random.randint(0, len(ptc_list[t - 1]))
            ptc_list[t - 1] = np.delete(ptc_list[t - 1], who)

        for i in range(k):    # add
            ptc_list[t - 1] = np.append(arr=ptc_list[t - 1], values=np.random.randint(0, D))



        # simulation
        curr_change = np.array([np.random.choice([-stepsize, stepsize], p=[1 - p_i, p_i]) for p_i in probabilities[t - 1]])
        ptc_list[t] = ptc_list[t - 1] + curr_change


        # calculating theta and delta
        delta[t] = dc(N_L, N_0)
        theta_list[t] = theta(t, delta[t], theta_list)   # updating theta(t) list

        # calculating concentrations and probabilities
        for j in range(n_bin):
            where = np.where(np.logical_and(ptc_list[t - 1] >= bins_list[j], ptc_list[t - 1] < bins_list[j + 1]))[0]
            c_list[t, j] = len(where) / n
            if t <= tau_s:
                probabilities[t, where] = p(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])

            elif t > tau_s:
                probabilities[t, where] = p_no_signal(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p_no_signal(theta_list[t], thresh, c_list[t, j])


            ## new
            # p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])
            H[t, j] = np.heaviside(c_list[t, j] - thresh, 1)



        # checking the particles at the boundaries and making sure no one passes them.
        ptc_list[t][np.where(ptc_list[t - 1] < 0)] = 0
        ptc_list[t][np.where(ptc_list[t - 1] > D)] = D

        # note number of particles at 1st and last quarter
        N_L[t] = len(np.where(ptc_list[t-1] > int(3 * D / 4))[0])
        N_0[t] = len(np.where(ptc_list[t-1] < int(D / 4))[0])




    pd.DataFrame(ptc_list).to_csv("results/changing_signal_once/ptc_list.csv")   # save the array of particles' positions
    pd.DataFrame(delta).to_csv("results/changing_signal_once/delta.csv")   # save the delta list
    pd.DataFrame(theta_list).to_csv(f"results/changing_signal_once/theta.csv")   # save the list of angles

    return ptc_list, delta, theta_list



def RW_simulation_changing_signal_twice(n, t_f, stepsize, D, k, thresh, n_bin, tau_d):
    """a simulation of random walk of a group of particles, with decay. Starting with a signal of π/2 that lasts for tau_s
    time-frames, and then changes to 0 for a time period of tau_d time-frames, and then again another signal of π/2 that
     lasts for tau_s time-frames, and finally the signal returns to 0 for the rest of the simulation. This function
     returns an array composed of vectors of positions of all particle for each time frame.

    :parameter tau_d the time delay between the two signals
    :parameter n number of particles.
    :parameter D length of the cross-section of the stem.
    :parameter t_f number of time frames.
    :parameter thresh threshold concentration
    :parameter n_bin number of segments
    :parameter stepsize in x-axis, usually 1.
    :parameter k number of particles to kill and create at each iteration
    :returns: an array of locations of all particles, as a function of time."""

    # initial settings

    def p(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin((theta) - np.pi / 2)) / 2

    def p_no_signal(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin(theta)) / 2

    def theta(t, delta, theta_list):
        return delta * 0.005 + theta_list[t - 1]

    def dc(N_L, N_0):
        return ((N_L - N_0)[t - 1] + 0.0001) / ((N_L + N_0)[t - 1] + 0.0001)

    ptc_list = list(np.zeros(t_f))
    ptc_list[0] = np.random.randint(0, D, size=n, dtype=int)
    N_0, N_L = np.zeros(t_f), np.zeros(t_f)
    theta_list = np.zeros(t_f)
    delta = np.zeros(t_f)
    bins_list = np.linspace(0, D, n_bin + 1, dtype=int)  # setting up the bins list
    c_list = np.zeros((t_f, n_bin))  # setting the concentrations list
    probabilities = np.full((t_f, n), 0.5)  # setting the probabilities array
    p_list, H = np.full((t_f, n_bin), 0.5), np.zeros((t_f, n_bin))

    for t in range(1, t_f):

        # adding / removing particles. for now, k particle at a time.
        for i in range(k):  # remove
            who = np.random.randint(0, len(ptc_list[t - 1]))
            ptc_list[t - 1] = np.delete(ptc_list[t - 1], who)

        for i in range(k):  # add
            ptc_list[t - 1] = np.append(arr=ptc_list[t - 1], values=np.random.randint(0, D))

        # simulation
        curr_change = np.array([np.random.choice([-stepsize, stepsize], p=[1 - p_i, p_i]) for p_i in probabilities[t - 1]])
        ptc_list[t] = ptc_list[t - 1] + curr_change

        # calculating theta and delta
        delta[t] = dc(N_L, N_0)
        theta_list[t] = theta(t, delta[t], theta_list)  # updating theta(t) list

        # calculating concentrations and probabilities
        for j in range(n_bin):
            where = np.where(np.logical_and(ptc_list[t - 1] >= bins_list[j], ptc_list[t - 1] < bins_list[j + 1]))[0]
            c_list[t, j] = len(where) / n

            if t <= 100 or 100 + tau_d <= t < 200 + tau_d:
                probabilities[t, where] = p(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])


            elif t >= 200 + tau_d or 100 < t < 100 + tau_d:
                probabilities[t, where] = p_no_signal(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p_no_signal(theta_list[t], thresh, c_list[t, j])

            ## new
            # p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])
            H[t, j] = np.heaviside(c_list[t, j] - thresh, 1)

        # checking the particles at the boundaries and making sure no one passes them.
        ptc_list[t][np.where(ptc_list[t - 1] < 0)] = 0
        ptc_list[t][np.where(ptc_list[t - 1] > D)] = D

        # note number of particles at 1st and last quarter
        N_L[t] = len(np.where(ptc_list[t - 1] > int(3 * D / 4))[0])
        N_0[t] = len(np.where(ptc_list[t - 1] < int(D / 4))[0])

    pd.DataFrame(ptc_list).to_csv("results/changing_signal_twice/ptc_list.csv")   # save the array of particles' positions
    pd.DataFrame(delta).to_csv("results/changing_signal_twice/delta.csv")   # save the delta list
    pd.DataFrame(theta_list).to_csv(f"results/changing_signal_twice/theta.csv")   # save the list of angles

    return ptc_list, delta, theta_list



def delta_max_tau_d(t_f, tau_d_list):
    """this function finds the maximum values of the dθ/dt and normalises them by the reference extrema.
    the function takes a list of tau_d values, and uploads the simulations which were generated with those tau_d's.
    notice that the duration of the signals is 100 time-frames.
    :param t_f number of time frames.
    :param tau_d_list list of tau_d values (list of the tau_d's which were used to generate the simulations).
    :return the normalised delta_max and their time"""

    peaks, time, i = np.zeros(len(tau_d_list)), np.zeros(len(tau_d_list)), 0

    # find the extrema of the reference
    theta_ref = np.array(pd.read_csv(f"data/changing_tau_d/theta_ref.csv"))[:, 1]
    d_theta_ref = np.array([(theta_ref[t + 1] - theta_ref[t]) / 0.005 for t in range(t_f - 1)])
    delta_ref = max(d_theta_ref[100: t_f + 1])

    for tau_d in tau_d_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_d/theta_{tau_d}.csv"))[:, 1]
        d_theta = np.array([(theta[t + 1] - theta[t]) / 0.005 for t in range(t_f - 1)])
        peak = max(d_theta[tau_d + 100: t_f + 1])
        peaks[i] = peak / delta_ref
        time[i] = tau_d
        i += 1

    return time, peaks


def delta_max_tau_s(t_f, tau_s_list):
    """this function finds the maximum values of the dθ/dt. the function takes a list of tau_s values, and uploads the
    simulations which were generated with those tau_s's.
    :param t_f number of time frames.
    :param tau_s_list list of tau_s values (list of the tau_s's which were used to generate the simulations).
    :return delta_max and their time"""

    peaks, time, i = np.zeros(len(tau_s_list)), np.zeros(len(tau_s_list)), 0

    for tau_s in tau_s_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_s/theta_{tau_s}.csv"))[:, 1]
        d_theta = np.array([(theta[t + 1] - theta[t]) / 0.005 for t in range(t_f - 1)])
        peak = max(d_theta[tau_s + 100: t_f + 1])
        peaks[i] = peak
        time[i] = tau_s + 100
        i += 1

    return time, peaks


def theta_max_tau_d(t_f, tau_d_list):
    """this function finds the maximum values of θ and normalizes them by the reference extrema.
    :param t_f number of time frames.
    :param tau_d_list list of tau_d values (list of the tau_d's which were used to generate the simulations).
    :return the normalized theta_max and their time"""
    peaks, time, i = np.zeros(len(tau_d_list)), np.zeros(len(tau_d_list)), 0

    # find the extrema of the reference
    theta_ref = np.array(pd.read_csv(f"data/changing_tau_d/theta_ref.csv"))[:, 1]
    d_theta_ref = np.array([(theta_ref[t + 1] - theta_ref[t]) / 0.005 for t in range(t_f - 1)])
    delta_ref = max(d_theta_ref[100: t_f + 1])

    for tau_d in tau_d_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_d/theta_{tau_d}.csv"))[:, 1]
        peak = max(theta[tau_d + 100: t_f + 1])
        peaks[i] = peak / delta_ref
        time[i] = tau_d
        i += 1

    return time, peaks


def theta_max_tau_s(t_f, tau_s_list):
    """this function finds the maximum values of θ. the function takes a list of tau_s values, and uploads the simulations
     which were generated with those tau_s's.
    :param t_f number of time frames.
    :param tau_s_list list of tau_s values (list of the tau_s's which were used to generate the simulations).
    :return delta_max and their time"""

    peaks, time, i = np.zeros(len(tau_s_list)), np.zeros(len(tau_s_list)), 0

    for tau_s in tau_s_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_s/theta_{tau_s}.csv"))[:, 1]
        peak = max(theta[tau_s: t_f + 1])
        peaks[i] = peak
        time[i] = tau_s
        i += 1

    return time, peaks



### automation
def RW_simulation_changing_signal_once_multi(n, t_f, stepsize, D, k, thresh, n_bin, tau_s, index):
    """a simulation of random walk of a group of particles, with decay and a varying angle. we just check the behaviour of the particles
    and connect to the angle. we have 1 varying signal that lasts tau_s seconds.

    :parameter tau_s the duration of the signal
    :parameter n No. of particles
    :parameter t_f No. of time frames
    :parameter thresh threshold concentration
    :parameter n_bin number of segments
    :parameter stepsize in x-axis, usually 1.
    :parameter D range of the initial spread (further boundary)
    :parameter k number of particles to kill and create at each iteration
    :returns: an array of locations of all particles, as a function of time."""


    # initial settings

    def p(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin(theta - np.pi / 2)) / 2


    def p_no_signal(theta, thresh, c_j):
        return np.heaviside(c_j - thresh, 1) - np.sign(c_j - thresh) * (1 - np.sin(theta)) / 2

    def theta(t, delta, theta_list):
        return delta * 0.005 + theta_list[t - 1]


    def dc(N_L, N_0):
        return ((N_L - N_0)[t - 1] + 0.0001) / ((N_L + N_0)[t - 1] + 0.0001)


    ptc_list = list(np.zeros(t_f))
    ptc_list[0] = np.random.randint(0, D, size=n, dtype=int)
    N_0, N_L = np.zeros(t_f), np.zeros(t_f)
    theta_list = np.zeros(t_f)
    delta = np.zeros(t_f)
    bins_list = np.linspace(0, D, n_bin + 1, dtype=int)    # setting up the bins list
    c_list = np.zeros((t_f, n_bin))   # setting the concentrations list
    probabilities = np.full((t_f, n), 0.5)    # setting the probabilities array
    p_list, H = np.full((t_f, n_bin), 0.5), np.zeros((t_f, n_bin))



    for t in range(1, t_f):

        # adding / removing particles. for now, k particle at a time.
        for i in range(k):    # remove
            who = np.random.randint(0, len(ptc_list[t - 1]))
            ptc_list[t - 1] = np.delete(ptc_list[t - 1], who)

        for i in range(k):    # add
            ptc_list[t - 1] = np.append(arr=ptc_list[t - 1], values=np.random.randint(0, D))



        # simulation
        curr_change = np.array([np.random.choice([-stepsize, stepsize], p=[1 - p_i, p_i]) for p_i in probabilities[t - 1]])
        ptc_list[t] = ptc_list[t - 1] + curr_change


        # calculating theta and delta
        delta[t] = dc(N_L, N_0)
        theta_list[t] = theta(t, delta[t], theta_list)   # updating theta(t) list

        # calculating concentrations and probabilities
        for j in range(n_bin):
            where = np.where(np.logical_and(ptc_list[t - 1] >= bins_list[j], ptc_list[t - 1] < bins_list[j + 1]))[0]
            c_list[t, j] = len(where) / n
            if t <= tau_s:
                probabilities[t, where] = p(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])

            elif t > tau_s:
                probabilities[t, where] = p_no_signal(theta_list[t], thresh, c_list[t, j])
                p_list[t, j] = p_no_signal(theta_list[t], thresh, c_list[t, j])


            ## new
            # p_list[t, j] = p(theta_list[t], thresh, c_list[t, j])
            H[t, j] = np.heaviside(c_list[t, j] - thresh, 1)



        # checking the particles at the boundaries and making sure no one passes them.
        ptc_list[t][np.where(ptc_list[t - 1] < 0)] = 0
        ptc_list[t][np.where(ptc_list[t - 1] > D)] = D

        # note number of particles at 1st and last quarter
        N_L[t] = len(np.where(ptc_list[t-1] > int(3 * D / 4))[0])
        N_0[t] = len(np.where(ptc_list[t-1] < int(D / 4))[0])



    pd.DataFrame(ptc_list).to_csv(f"C:/Users/Noa/Desktop/changing_c_0/thetas_26.75/ptc_list_{index}.csv")   # save the 1st list of particles
    pd.DataFrame(delta).to_csv(f"C:/Users/Noa/Desktop/changing_c_0/thetas_26.75/delta_{index}.csv")   # save the delta list
    pd.DataFrame(theta_list).to_csv(f"C:/Users/Noa/Desktop/changing_c_0/thetas_26.75/theta_{index}.csv")   # save the list of angles

    # pd.DataFrame(ptc_list).to_csv(f"C:/Users/Noa/Desktop/{D}/ptc_list.csv")   # save the 1st list of particles
    # pd.DataFrame(delta).to_csv(f"C:/Users/Noa/Desktop/{D}/delta.csv")   # save the delta list
    # pd.DataFrame(theta_list).to_csv(f"C:/Users/Noa/Desktop/{D}/theta.csv")   # save the list of angles


    return ptc_list, delta, theta_list