from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np
import math

# Assume that working with right plane function y=sqrt(x**7)

# DEFINE FUNCTION AND PLOT PARAMETERS #

lower_bound = 0
upper_bound = 1
T = 2 * (upper_bound - lower_bound)  # since the function is periodical, after reflection.
L = T / 2
frequency = 2 * math.pi / T
N = 32  # Upper series bound

odd_spectrum = dict()
odd_linear_modified_spectrum = dict()


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
    elif param == "b_k":
        for amplitude_val in dot:
            series_sum = 0
            odd_spectrum[0] = coefficient[0]
            for i in range(1, max_n):
                freq = i * frequency * amplitude_val
                series_sum += coefficient[i] * math.sin(freq)
                odd_spectrum[freq] = odd_spectrum.get(freq, 0) + coefficient[i]
            res.append(series_sum)
    elif param == "c_k":
        for amplitude_val in dot:
            series_sum = 0
            odd_linear_modified_spectrum[0] = coefficient[0]
            for i in range(1, max_n):
                freq = i * frequency * amplitude_val
                series_sum += coefficient[i] * math.sin(freq)
                odd_linear_modified_spectrum[freq] = odd_linear_modified_spectrum.get(freq, 0) + coefficient[i]
            res.append(series_sum)
    return res


def visualize_graph(coefficient, graph_type, fig, pos, function=None):
    incr = 2 / N
    graph_name = None
    if graph_type != "d_k" and graph_type != "e_k":
        x_value = np.arange(-upper_bound, upper_bound + incr, incr)
        res = list()
        for val in range(0, len(x_value)):
            res.append(function(x_value[val]))
        if graph_type == "a_k":
            graph_name = "Odd Fourier Approximation"
        elif graph_type == "b_k":
            graph_name = "Even Fourier Approximation"
        elif graph_type == "c_k":
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
    else:
        if graph_type == "d_k":
            init_axis(fig, pos + 1, "Spectrum of odd and linear mod with the same amplitude", np.arange(-100, 100, 5),
                      "Frequency", "Frequency")
            plt.scatter(odd_spectrum.keys(), odd_spectrum.keys(), color="yellow")
            plt.scatter(odd_linear_modified_spectrum.keys(), odd_linear_modified_spectrum.keys(), color="red",
                        marker='.')
        else:
            init_axis(fig, pos + 1, "Spectrum of odd and linear mod with different amplitude", np.arange(-100, 100, 5))
            plt.scatter(odd_spectrum.keys(), odd_spectrum.values(), color="yellow")
            plt.scatter(odd_linear_modified_spectrum.keys(), odd_linear_modified_spectrum.values(), color="red",
                        marker='.')


def get_even_reflection_function(x, n):
    return get_specific_function(x) * np.cos(n * x * frequency)


def get_odd_reflection_function(x, n):
    return get_specific_function(x) * np.sin(n * x * frequency)


def get_linear_modified_function_odd(x, n):
    return get_specific_odd_function(x) - get_specific_odd_function(lower_bound) - \
           (get_specific_odd_function(upper_bound) - get_specific_odd_function(lower_bound)) * x * np.sin(
        n * x * frequency) / upper_bound


def get_coefficient(n, coefficient_type):
    if coefficient_type == "a_k":
        return 2 * integrate.quad(get_even_reflection_function, lower_bound, upper_bound, args=n)[0] / L
    elif coefficient_type == "b_k":
        return 2 * integrate.quad(get_odd_reflection_function, lower_bound, upper_bound, args=n)[0] / L
    elif coefficient_type == "c_k":
        return 2 * integrate.quad(get_linear_modified_function_odd, lower_bound, upper_bound, args=n)[0] / L


def get_specific_function(x):
    return math.sqrt(x ** 7)
    # return math.sin(4*x) + math.sin(3*x)

# This is the experiment with different function.
# Note, that there are many spectrum in fourier series, due I'm not expansion to natural number.
# So, in this cases there are wouldn't be 3 and 4 rectangle peak.


def get_specific_even_function(x):
    return get_specific_function(x) if x > 0 else get_specific_function(-x)


def get_specific_odd_function(x):
    return get_specific_function(x) if x > 0 else -get_specific_function(-x)


def get_linear_modified_function(x):
    return get_specific_odd_function(x) - get_specific_odd_function(lower_bound) - \
           (get_specific_odd_function(upper_bound) - get_specific_odd_function(lower_bound)) * x / upper_bound


def main():
    figure = plt.figure(figsize=(30, 20))
    coefficient_range = np.arange(0, N, 1)
    pos = 420
    coefficient = ["a_k", "b_k", "c_k"]
    function = [get_specific_even_function, get_specific_odd_function, get_linear_modified_function]
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

    # Now we have expansion into a fourier series our function by the spectrum: frequency * n * t.
    # Let's check that if we linear modify our function - nothing has changed.
    # The idea is to create dictionary [frequency] - [coefficient]
    # And then just plot graphically [frequency] - [frequency] for linear modified and usual function.
    # Also, if we plot graph [frequency] - [coefficient] then we will see the spectrum of other function.
    # The graph in this case will be different, due to different coefficient in cos/sin.

    # I will call this graph d_k and e_k
    visualize_graph(store, "d_k", figure, pos)
    pos += 1
    visualize_graph(store, "e_k", figure, pos)
    # Now (d_k) we can see that linear transform accelerate the convergence. And didn't modify the spectrum.
    # e_k show that if we project the point on the x axis then it will be the same spectrum.
    plt.show()


main()
