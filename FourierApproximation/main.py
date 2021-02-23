from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import math

# Assume that working with right plane semicircle y=sqrt(x**7)

# DEFINE SEMICIRCLE AND PLOT PARAMETERS #

lower_bound = 0
upper_bound = 1
T = 2 * (upper_bound - lower_bound)  # since the function is periodical, after reflection.
L = T / 2
frequency = 2 * math.pi / T
N = 55  # Upper series bound


def init_axis(figure, pos, title_name, max_tick_number=np.arange(0, N, 1), x_name='Frequency', y_name='Amplitude'):
    axis = figure.add_subplot(pos)
    figure.add_axes(axis)
    axis.title.set_text(title_name)
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    axis.set_xticks(max_tick_number)


def visualize_coefficient(amplitude_value, coefficient_range, fig, pos, title_name):
    init_axis(fig, pos + 1, title_name, x_name="Relative frequency, 1 / w")
    plt.scatter(coefficient_range, amplitude_value, color="green")
    plt.plot(coefficient_range, amplitude_value)


def get_fourier_approximation(coefficient, dot, param, max_n=N):
    res = list()
    assert (max_n <= N)
    if param == "a_k":
        for amplitude_val in dot:
            series_sum = coefficient[0] / 2
            for i in range(1, max_n):
                series_sum += coefficient[i] * math.cos(i * frequency * amplitude_val)
            res.append(series_sum)
    elif param == "b_k" or param == "c_k":
        for amplitude_val in dot:
            series_sum = 0
            for i in range(1, max_n):
                series_sum += coefficient[i] * math.sin(i * frequency * amplitude_val)
            res.append(series_sum)
    return res


def visualize_graph(coefficient, graph_type, fig, pos, function):
    incr = 2 / N
    x_value = np.arange(-upper_bound, upper_bound + incr, incr)
    res = list()
    for val in range(0, len(x_value)):
        res.append(function(x_value[val]))
    if graph_type == "a_k":
        graph_name = "Odd Fourier Approximation"
    elif graph_type == "b_k":
        graph_name = "Even Fourier Approximation"
    else:
        graph_name = "Odd Fourier Approximation modified"
    init_axis(fig, pos + 1, graph_name, np.arange(-upper_bound, upper_bound + 0.1, 0.1))

    # Draw approximation
    fourier = get_fourier_approximation(coefficient, x_value, graph_type, int(N / 2))
    x_new = np.linspace(x_value.min(), x_value.max(), 500)
    plt.plot(x_new, interp1d(x_value, fourier, kind='quadratic')(x_new), label="Fourier approximation")
    # plt.scatter(x_new, fourier, color="green")

    # Draw origin
    plt.scatter(x_value, res, color="yellow")
    plt.plot(x_value, res, color='red', label="Origin function")


def get_even_reflection_semicircle(x, n):
    return get_semicircle_function(x) * np.cos(n * x * frequency)


def get_odd_reflection_semicircle(x, n):
    return get_semicircle_function(x) * np.sin(n * x * frequency)


def get_linear_modified_semicircle(x, n):
    return get_semicircle_odd_function(x) - get_semicircle_odd_function(lower_bound) - \
           (get_semicircle_odd_function(upper_bound) - get_semicircle_odd_function(lower_bound)) * x * np.sin(
        n * x * frequency) / upper_bound


def get_coefficient(n, coefficient_type):
    if coefficient_type == "a_k":
        return 2 * integrate.quad(get_even_reflection_semicircle, lower_bound, upper_bound, args=n)[0] / L
    elif coefficient_type == "b_k":
        return 2 * integrate.quad(get_odd_reflection_semicircle, lower_bound, upper_bound, args=n)[0] / L
    elif coefficient_type == "c_k":
        return 2 * integrate.quad(get_linear_modified_semicircle, lower_bound, upper_bound, args=n)[0] / L


def get_semicircle_function(x):
    return math.sqrt(x ** 7)


def get_semicircle_even_function(x):
    return np.sqrt(x ** 7) if x > 0 else np.sqrt(-x ** 7)


def get_semicircle_odd_function(x):
    return np.sqrt(x ** 7) if x > 0 else -np.sqrt(-x ** 7)


def get_linear_modified_function(x):
    return get_semicircle_odd_function(x) - get_semicircle_odd_function(lower_bound) - \
           (get_semicircle_odd_function(upper_bound) - get_semicircle_odd_function(lower_bound)) * x / upper_bound


def main():
    figure = plt.figure(figsize=(30, 20))
    coefficient_range = np.arange(0, N, 1)
    pos = 420
    coefficient = ["a_k", "b_k", "c_k"]
    function = [get_semicircle_even_function, get_semicircle_odd_function, get_linear_modified_function]
    store = list()
    w = 0
    for c in coefficient:
        for i in range(0, N):
            store.append(get_coefficient(i, c))
        visualize_coefficient(store, coefficient_range, figure, pos, str(c))
        pos += 1
        visualize_graph(store, str(c), figure, pos, function[w])
        pos += 1
        store.clear()
        w += 1
    # Here we can see, that even reflection converges faster than odd. (1/k**2 vs 1/k)
    # This is relative frequency in x axis, the spectrum should be 1w,2w..etc

    # Here we have modified odd by adding linear function: (w=2) and what next? should ask.
    plt.show()


main()
