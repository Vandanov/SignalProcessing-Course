import matplotlib.pyplot as plt
import numpy as np
import math

# Assume that working with right plane semicircle x = f(y) = sqrt(r^2-(y+b)^2)

# DEFINE SEMICIRCLE AND PLOT PARAMETERS #
from scipy import integrate

radius = 1
lower_bound = 0
upper_bound = 1
b = -radius  # to rise it, because then we need reflect that
# look like ) ) ) with period = sqrt(r) and above x axis


def init_axis(figure, pos, title_name, x_name='Frequency', y_name='Amplitude'):
    axis = figure.add_subplot(pos)
    figure.add_axes(axis)
    axis.title.set_text(title_name)
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)


def generate_example(amplitude_value, range_, fig, pos, forward_name):
    frequency_value = np.arange(range_[0], range_[1], range_[2])

    init_axis(fig, pos + 1, forward_name)
    plt.scatter(frequency_value, amplitude_value, color="green")
    plt.plot(frequency_value, amplitude_value)


def rectangle_impulse_example(fig, range_, pos):
    time_value = np.arange(range_[0], range_[1], range_[2])
    val = int((range_[1] - range_[0]) / range_[2])
    width = 1
    amplitude_value = np.zeros((val,))

    from_ = int(val / 2 - math.ceil(width / 2 / range_[2]))
    to_ = int(val / 2 + math.ceil(width / 2 / range_[2]))
    amplitude_value[from_:to_] = 1
    generate_example(amplitude_value, time_value, fig, pos,
                     forward_name='Rectangle Signal Forward')


def get_even_reflection_semicircle(y, n):
    return get_semicircle_function(y) * np.cos(n * y)


def get_odd_reflection_semicircle(y, n):
    return get_semicircle_function(y) * np.sin(n * y)


def get_coefficient(n, coefficient_type):
    if coefficient_type == "a_k":
        return 2 * integrate.quad(get_even_reflection_semicircle, lower_bound, upper_bound, args=n)[0]
    elif coefficient_type == "b_k":
        return 2 * integrate.quad(get_odd_reflection_semicircle, lower_bound, upper_bound, args=n)[0]


def get_semicircle_function(y):
    return math.sqrt(radius ** 2 - (y + b) ** 2)

def main():
    figure = plt.figure(figsize=(35, 25))
    range_ = [0, 1, 0.01]
    pos = 420
    max_val = int((range_[1] - range_[0]) / range_[2]) + 1
    coefficient = ["a_k", "b_k"]
    store = list()
    for c in coefficient:
        for i in range(1, max_val):
            store.append(get_coefficient(i, c))
        generate_example(store, range_, figure, pos, str(c))
        pos += 2
        store.clear()
    # Here we can see, that even reflection converges faster than odd. (1/k**2 vs 1/k)
    # This is relative frequency in x axis

    plt.show()


main()
