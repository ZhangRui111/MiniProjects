import numpy as np
import matplotlib.pyplot as plt


def plot_steps(steps_episodes, save_path=None):
    length = len(steps_episodes)
    x_axis_data = np.arange(0, length)
    y_axis_data = np.asarray(steps_episodes)

    plt.ion()
    plt.plot(x_axis_data, y_axis_data)
    plt.xlabel('episode')
    plt.ylabel('steps per episode')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_multi_lines(all_data, all_labels, save_path=None):
    size = len(all_data)
    size_labels = len(all_labels)
    assert size == size_labels

    plt.ion()
    for i in range(size):
        data = all_data[i]
        length = len(data)
        x_axis_data = np.arange(0, length)
        y_axis_data = np.asarray(data)
        plt.plot(x_axis_data, y_axis_data, label=all_labels[i])
    plt.xlabel('episode')
    plt.ylabel('steps per episode')
    plt.legend(loc='best')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_smooth_steps(steps_episodes, save_path=None):
    all_lines = []
    all_smooth_lines = []
    all_configs = []

    running_avg = running_average(steps_episodes, interval=50)
    smoothen_running_avg = smoothen_line(convolve_line(running_avg, interval=100))
    all_smooth_lines.append(smoothen_running_avg)
    all_configs.append({'alpha': 0.2})
    all_lines.append(running_avg[:-1 * 200])  # strip the tail due to smoothen

    plt.ion()
    # plot the normal line
    p = plt.plot(all_lines[0], alpha=all_configs[0]['alpha'])
    color = p[-1].get_color()
    # plot the smooth line
    plt.plot(all_smooth_lines[0][0], all_smooth_lines[0][1], color=color)

    plt.xlabel('episode')
    plt.ylabel('steps per episode')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def average_from_start(data):
    """Calculate the average from start. AVG([ 0 ~ i ])"""
    accum = 0
    avg = []
    for idx in range(len(data)):
        accum += data[idx]
        avg.append(accum / (idx + 1 + 1e-7))
    return avg


def running_average(data, interval=50):
    """Calculate the running average of given data. AVG([ i ~ i+interval ])"""
    avg = []
    for i in range(len(data) - interval):
        left = i
        right = min(i + interval, len(data))
        avg.append(sum(data[left:right]) / (right - left))
    return avg


def convolve_line(data, interval=200):
    """Convolve on an array of data, with interval specified as a parameter"""
    return np.convolve(
        data,
        np.ones((interval,)) / interval, mode='same'
    )[:-1 * interval]


def smoothen_line(data, N=300):
    """Smoothen a line by interpolation."""
    from scipy.interpolate import make_interp_spline, BSpline
    xold = range(len(data))
    xnew = np.linspace(xold[0], xold[-1], N)  # 300 represents number of points to make between T.min and T.max
    spl = make_interp_spline(xold, data, k=3)  # BSpline object
    power_smooth = spl(xnew)
    return xnew, power_smooth
