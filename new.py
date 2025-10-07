import numpy as np
import matplotlib.pyplot as plt

def read_sequence_from_file(filename):
    with open(filename, 'r') as f:
        line = f.readline().strip()
    sequence = [int(ch) for ch in line if ch in ('0', '1')]
    return sequence

def spectral_analysis(sequence):
    x = np.array(sequence)*2 - 1
    n = len(x)
    fft_vals = np.fft.fft(x)
    fft_vals = fft_vals[:n//2]
    
    amplitudes = np.abs(fft_vals)/n
    freqs = np.fft.fftfreq(n)[:n//2]
    
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, amplitudes)
    plt.title('Спектральный анализ псевдослучайной последовательности')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

# Пример использования
filename = '9x_8_17496.txt'
sequence = read_sequence_from_file(filename)
spectral_analysis(sequence)
