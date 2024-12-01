import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def plot_RW_simulation_frames(t_f, D, ptc_list):
    """this function plots a few relevant frames from the simulation"""

    b = np.linspace(0, D, int(D/10))

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].hist(ptc_list[0], bins=b, width=2)
    axis[0, 0].set_title("t=0")
    axis[0, 0].set_xlim(-10, D + 5)

    axis[0, 1].hist(ptc_list[int(t_f / 20)], bins=b, width=2)
    axis[0, 1].set_title(f"t={int(t_f / 20)}")
    axis[0, 1].set_xlim(-10, D + 5)

    axis[1, 0].hist(ptc_list[int(t_f / 4)], bins=b, width=2)
    axis[1, 0].set_title(f"t={int(t_f / 4)}")
    axis[1, 0].set_xlim(-10, D + 5)

    axis[1, 1].hist(ptc_list[int(t_f / 3)], bins=b, width=2)
    axis[1, 1].set_title(f"t={int(t_f / 3)}")
    axis[1, 1].set_xlim(-10, D + 5)

    plt.show()

    figure, axis = plt.subplots(2, 2)

    axis[0, 0].hist(ptc_list[int(t_f / 10)], bins=b, width=2)
    axis[0, 0].set_title(f"t={int(t_f / 10)}")
    axis[0, 0].set_xlim(-10, D + 5)

    axis[0, 1].hist(ptc_list[int(t_f / 6)], bins=b, width=1.5)
    axis[0, 1].set_title(f"t={int(t_f / 6)}")
    axis[0, 1].set_xlim(-10, D + 5)

    axis[1, 0].hist(ptc_list[int(t_f / 2)], bins=b, width=2)
    axis[1, 0].set_title(f"t={int(t_f / 2)}")
    axis[1, 0].set_xlim(-10, D + 5)

    axis[1, 1].hist(ptc_list[t_f - 1], bins=b, width=2)
    axis[1, 1].set_title(f"t={t_f}")
    axis[1, 1].set_xlim(-10, D + 5)

    plt.show()


def plot_theta_dc(t_f, theta_list, delta, thresh, n_bin):
    """this function plots theta and delta as a function of time.
    :param t_f number of time frames.
    :param theta_list list of the angle theta.
    :param delta list of delta.
    :param thresh threshold concentration.
    """
    t = np.arange(0, t_f, 1)
    dc = delta[:, 1]

    plt.plot(t, np.degrees(theta_list[:, 1]))
    plt.title(f"θ as a function of time\n {thresh} threshold, {n_bin} segments")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.show()

    plt.plot(t, dc)
    plt.title(f"∆ as a function of time\n{thresh} threshold, {n_bin} segments")
    plt.xlabel("time"), plt.ylabel("∆")
    plt.show()



def plot_C_p(C, p_list, n_bin, t_f):
    """this function plots """
    t = np.arange(0, t_f, 1)

    for j in range(1, n_bin+1):
        plt.plot(t, C[:, j], '.', label=f"{j}")
    plt.title("C as a function of time")
    plt.legend()
    plt.show()

    for j in range(1, n_bin+1):
        plt.plot(t, p_list[:, j], '.', label=f"{j}")
    plt.title("probability as a function of time")
    plt.legend()
    plt.show()



def plot_theta_legend_tau_d(t_f, tau_d_list):
    t, i = np.arange(0, t_f, 1), 0
    colors = plt.cm.hsv(np.linspace(0, 1, len(tau_d_list) + 1))
    for tau_d in tau_d_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_d/theta_{tau_d}.csv"))
        plt.plot(t, np.degrees(theta[:, 1]), label=f"τ={tau_d}", color=colors[i])
        i += 1

    plt.title("θ as a function of time for different delays τ_d")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.legend()
    plt.show()


def plot_theta_legend_tau_s(t_f, tau_s_list):
    t, i = np.arange(0, t_f, 1), 0
    colors = plt.cm.hsv(np.linspace(0, 1, len(tau_s_list) + 1))
    for tau_s in tau_s_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_s/theta_{tau_s}.csv"))
        plt.plot(t, np.degrees(theta[:, 1]), label=f"τ={tau_s}", color=colors[i])
        i += 1

    plt.title("θ as a function of time for different delays τ_s")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.legend()
    plt.show()


def plot_delta_peaks(t_f, iterations):
    peaks, time = np.zeros(iterations), np.zeros(iterations)
    for i in range(iterations):
        delta_i = np.array(pd.read_csv(f"C:/Users/Noa/Desktop/delta_{i}.csv"))[:, 1]
        peak = max(delta_i[i * 100 + 100 : t_f + 1])
        peaks[i] = peak
        time[i] = np.where(delta_i == peak)[0][0]

    plt.plot(time, np.degrees(peaks), ".")
    plt.title(f"maxima of ∆ as a function of time")
    plt.xlabel("time"), plt.ylabel("∆_max")
    plt.legend()
    plt.show()



def plot_theta_vs_derivative(t_f):
    t = np.arange(0, t_f, 1)
    dt = np.arange(0, t_f - 1, 1)

    theta = np.array(pd.read_csv(f"results/theta.csv"))[:, 1]
    d_theta = np.array([(theta[i + 1] - theta[i]) / 0.005 for i in range(t_f - 1)])

    plt.title("θ vs ∂θ/∂t as a function of time")
    plt.plot(t, np.degrees(theta), label="θ")
    plt.plot(dt, np.degrees(d_theta), label="∂θ/∂t")
    plt.legend()
    plt.show()




def plot_theta_legend_c_0(t_f, c_0_list):
    t, i = np.arange(0, t_f, 1), 0
    colors = plt.cm.hsv(np.linspace(0, 1, len(c_0_list) + 2))

    for c_0 in c_0_list:
        theta = np.array(pd.read_csv(f"data/changing_c_0/theta_{c_0}.csv"))
        plt.plot(t, np.degrees(theta[:, 1]), label=f"c_0={c_0}", color=colors[i])
        i += 1

    plt.title("θ as a function of time for different threshold concentrations")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.legend()
    plt.show()


def plot_theta_legend_k(t_f, k_list):
    t, i = np.arange(0, t_f, 1), 0
    colors = plt.cm.hsv(np.linspace(0, 1, len(k_list) + 2))

    for k in k_list:
        theta_i = np.array(pd.read_csv(f"data/changing_k/theta_{k}.csv"))
        plt.plot(t, np.degrees(theta_i[:, 1]), label=f"k={k}", color=colors[i])
        i += 1

    plt.title("θ as a function of time for different values of k")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.legend()
    plt.show()


def plot_theta_legend_D(t_f, D_list):
    t, i = np.arange(0, t_f, 1), 0
    colors = plt.cm.hsv(np.linspace(0, 1, len(D_list) + 2))

    for D in D_list:
        theta_i = np.array(pd.read_csv(f"data/changing_D/theta_{D}.csv"))
        plt.plot(t, np.degrees(theta_i[:, 1]), label=f"D={D}", color=colors[i])
        i += 1

    plt.title("θ as a function of time for different lengths D")
    plt.xlabel("time"), plt.ylabel("θ")
    plt.legend()
    plt.show()


def plot_delta_max_tau_d(time, peaks):
    plt.plot(time, peaks)
    plt.plot(time, peaks, '.')
    plt.title("maxima of ∆ as a function of τ_d")
    plt.xlabel("τ_d"), plt.ylabel("∆_max / ∆_ref (deg)")
    plt.show()


def plot_delta_max_tau_s(time, peaks):
    plt.plot(time, peaks)
    plt.plot(time, peaks, '.')
    plt.legend()
    plt.title("maxima of ∆ as a function of τ_s")
    plt.xlabel("τ_s"), plt.ylabel("∆_max (deg)")
    plt.show()


def plot_derivatives_tau_d(t_f, tau_d_list):
    t, i = np.arange(0, t_f, 1), 0
    dt = np.arange(0, t_f - 1, 1)
    colors = plt.cm.hsv(np.linspace(0, 1, len(tau_d_list) + 1))

    for tau_d in tau_d_list:
        theta = np.array(pd.read_csv(f"data/changing_tau_d/theta_{tau_d}.csv"))[:, 1]
        d_theta = np.array([(theta[t + 1] - theta[t]) / 0.005 for t in range(t_f - 1)])
        smoothed_ytheta = np.convolve(d_theta, np.ones(30) / 30, mode='valid')
        smoothed_xtheta = dt[:len(smoothed_ytheta)]
        plt.plot(smoothed_xtheta, np.degrees(smoothed_ytheta), label=f"τ_d={tau_d}", color=colors[i])
        i += 1

    plt.title("∂θ/∂t as a function of time for different τ_d values")
    plt.ylabel("∂θ/∂t"), plt.xlabel("time")
    plt.legend()
    plt.show()


def plot_theta_max_tau_d(time, peaks):
    plt.plot(time, peaks)
    plt.plot(time, peaks, '.')
    plt.legend()
    plt.title("maxima of θ as a function of τ_d")
    plt.xlabel("τ_d"), plt.ylabel("θ_max / ∆_ref (deg)")
    plt.show()


def plot_theta_max_tau_s(time, peaks):
    plt.plot(time, peaks)
    plt.plot(time, peaks, '.')
    plt.legend()
    plt.title("maxima of θ as a function of τ_s")
    plt.xlabel("τ_s"), plt.ylabel("θ_max (deg)")
    plt.show()