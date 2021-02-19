import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.fft as fft


# 1) Пока меня нет в Новосибирске, пожалуйста, изучите самостоятельно тему «преобразование Фурье»
# и напишите программу, вычисляющую прямое и обратное преобразование Фурье,
# 2) использую формулу дискретного преобразования Фурье. Когда напишете функцию,
# 3) напишите программу, которая сравнивает ваше преобразование Фурье
# и результат быстрого преобразования Фурье из какой либо открытой библиотеки
# на различных сигналах: гармонический сигнал, случайный шум, прямоугольный импульс,
# сумма двух гармонических функций с разными частотами и амплитудами.
# 4) Результат вычисления преобразования Фурье покажите графически.
# Через неделю пришлю следующее задание.

# O(N^2) solution
def dft(sequence, element, is_forward=True):
    size = len(sequence)
    res = 0
    sign = pow(-1, int(is_forward))
    for i in range(0, size):
        res += sequence[i] * np.exp(2j * math.pi * i * element * sign / size)
    return res if is_forward else res / size


def fill_dft_array(sequence, is_forward=True):
    arr = []
    size = len(sequence)
    for i in range(0, size):
        arr.append(dft(sequence, i, is_forward))
    return arr


def init_axis(figure, pos, title_name, x_name='Time (seconds)', y_name='Amplitude'):
    axis = figure.add_subplot(pos)
    figure.add_axes(axis)
    axis.title.set_text(title_name)
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)


def generate_example(amplitude_value, time_value, fig, pos, forward_name, inverse_name):
    built_in_fourier_transform = fft.fft(amplitude_value)
    own_fourier_transform = fill_dft_array(amplitude_value, True)

    for i in range(1, 3):
        if i == 1:
            init_axis(fig, pos + i, forward_name)
            plt.scatter(time_value, np.abs(fill_dft_array(own_fourier_transform, False)), color="green")
            plt.plot(time_value, np.abs(fft.ifft(built_in_fourier_transform)))
            plt.scatter(time_value, np.abs(amplitude_value), color="red", marker=2)
        elif i == 2:
            init_axis(fig, pos + i, inverse_name)
            plt.plot(time_value, np.abs(built_in_fourier_transform))
            plt.scatter(time_value, np.abs(own_fourier_transform), color="green")


def harmonic_signal_example(fig, range_, pos):
    time_value = np.arange(range_[0], range_[1], range_[2])
    amplitude_value = np.sin(time_value)
    generate_example(amplitude_value, time_value, fig, pos,
                     forward_name='Harmonic Signal Forward',
                     inverse_name='Harmonic Signal Inverse')


def random_noise_example(fig, range_, pos):
    time_value = np.arange(range_[0], range_[1], range_[2])
    amplitude_value = np.random.normal(time_value)
    generate_example(amplitude_value, time_value, fig, pos,
                     forward_name='Noise Signal Forward',
                     inverse_name='Noise Signal Inverse')


def rectangle_impulse_example(fig, range_, pos):
    time_value = np.arange(range_[0], range_[1], range_[2])
    val = int((range_[1] - range_[0]) / range_[2])
    width = 1
    amplitude_value = np.zeros((val,))

    from_ = int(val / 2 - math.ceil(width / 2 / range_[2]))
    to_ = int(val / 2 + math.ceil(width / 2 / range_[2]))
    amplitude_value[from_:to_] = 1
    generate_example(amplitude_value, time_value, fig, pos,
                     forward_name='Rectangle Signal Forward',
                     inverse_name='Rectangle Signal Inverse')


def sum_different_harmonic_example(fig, range_, pos, ampl1, ampl2, freq1, freq2):
    time_value = np.arange(range_[0], range_[1], range_[2])
    amplitude_value = ampl1 * np.sin(time_value * freq1) + ampl2 * np.cos(time_value * freq2)
    generate_example(amplitude_value, time_value, fig, pos,
                     forward_name='Sum harmonic Forward, A1='
                                  + str(ampl1) + ';A2=' + str(ampl2)
                                  + ';freq1=' + str(freq1) + ';freq2' + str(freq2),
                     inverse_name='Sum harmonic Inverse the same param')


def main():
    figure = plt.figure(figsize=(35, 25))
    range_ = [-10, 10, 0.01]
    pos = 420
    vec = [harmonic_signal_example, random_noise_example, rectangle_impulse_example]

    for i in range(len(vec)):
        vec[i](figure, range_, pos)
        pos += 2
    sum_different_harmonic_example(figure, range_, pos, 2, 5, 1, 2)

    plt.show()


main()
