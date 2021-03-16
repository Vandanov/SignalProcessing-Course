import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.fft as fft


def convolve(lhs_function_val, rhs_function_val):
    out_len = len(lhs_function_val) + len(rhs_function_val) - 1

    lhs_extended = np.pad(lhs_function_val, (0, out_len - len(lhs_function_val)))
    rhs_extended = np.pad(rhs_function_val, (0, out_len - len(rhs_function_val)))
    res = list()
    for n in range(0, out_len):
        tmp_sum = 0
        for m in range(0, out_len):
            tmp_sum += lhs_extended[m] * rhs_extended[n - m]
        res.append(tmp_sum)
    return res


def convolve_via_fft(lhs_function_val, rhs_function_val):
    out_len = len(lhs_function_val) + len(rhs_function_val) - 1

    lhs_extended = np.pad(lhs_function_val, (0, out_len - len(lhs_function_val)))
    rhs_extended = np.pad(rhs_function_val, (0, out_len - len(rhs_function_val)))

    return np.fft.ifft(np.fft.fft(lhs_extended) * np.fft.fft(rhs_extended))


def correlate(lhs_function_val, rhs_function_val):
    out_len = len(lhs_function_val) + len(rhs_function_val) - 1

    lhs_extended = np.pad(lhs_function_val, (0, out_len - len(lhs_function_val)))
    rhs_extended = np.pad(rhs_function_val, (out_len - len(rhs_function_val), 0))
    res = []
    for n in range(0, out_len):
        tmp_sum = 0
        for m in range(0, out_len - n):
            tmp_sum += lhs_extended[m] * rhs_extended[n + m]
        res.append(tmp_sum)
    return res


def correlate_via_fft(lhs_function_val, rhs_function_val):
    out_len = len(lhs_function_val) + len(rhs_function_val) - 1

    lhs_extended = np.pad(lhs_function_val, (0, out_len - len(lhs_function_val)))
    rhs_extended = np.pad(rhs_function_val, (0, out_len - len(rhs_function_val)))

    fft1 = np.fft.fft(lhs_extended)
    fft2 = np.fft.fft(rhs_extended)
    return np.fft.fftshift(np.fft.ifft(np.conj(fft1) * fft2))


def init_axis(figure, pos, title_name, x_name='Time (seconds)', y_name='Amplitude'):
    axis = figure.add_subplot(pos)
    figure.add_axes(axis)
    axis.title.set_text(title_name)
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)


def generate_example(fig, pos, func_1, func_2, total_interval):
    time_extended = np.linspace(-1, 1, 2 * total_interval - 1)
    init_axis(fig, pos + 1, "Convolution")
    plt.scatter(time_extended, convolve(func_1, func_2), color="green", marker='s')
    plt.scatter(time_extended, convolve_via_fft(func_1, func_2).real, color="yellow", marker=2)
    plt.scatter(time_extended, np.convolve(func_1, func_2), color="blue", marker="o")

    init_axis(fig, pos + 2, "Correlation")
    plt.scatter(time_extended, correlate(func_1, func_2), color="green", marker='s')
    plt.scatter(time_extended, np.correlate(func_1, func_2, mode='full'), color="yellow", marker=2, label="original")
    plt.scatter(time_extended, correlate_via_fft(func_1, func_2).real, color="blue", marker=1)


def main():
    fig = plt.figure(figsize=(35, 25))
    pos = 420

    total_interval = 500
    t = np.linspace(0, 1, total_interval)
    func_1 = np.sin(2 * np.pi * 5 * t)
    func_2 = np.sin(2 * np.pi * 10 * t)
    generate_example(fig, pos, func_1, func_2, total_interval)

    pos += 2

    total_interval = 500
    t = np.linspace(0, 1, total_interval)
    func_1 = np.tan(2 * np.pi * 5 * t)
    func_2 = np.tan(2 * np.pi * 10 * t)
    generate_example(fig, pos, func_1, func_2, total_interval)

    pos += 2

    total_interval = 500
    t = np.linspace(0, 1, total_interval)
    func_1 = np.sin(2 * np.pi * 50 * t) + np.cos(2 * np.pi * 150 * t)
    func_2 = np.cos(2 * np.pi * 10 * t) - np.sin(np.sin(2 * np.pi * 50 * t))
    generate_example(fig, pos, func_1, func_2, total_interval)

    plt.show()


main()
