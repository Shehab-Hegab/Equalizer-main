import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def generate_sine_wave(duration, frequency, amplitude=1.0, sample_rate=44100):
    t = np.arange(0, duration, 1/sample_rate)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave

def create_wav_file(file_path, waves):
    samples = np.concatenate(waves)
    scaled_samples = np.int16(samples * 32767)
    write(file_path, 44100, scaled_samples)

def plot_fft(file_path, sample_rate=44100, max_frequency=1000):
    data = np.fromfile(file_path, dtype=np.int16)
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(fft_result), d=1/sample_rate)

    # Find the index corresponding to the max_frequency
    max_freq_index = int(len(frequencies) * max_frequency / (sample_rate / 2))

    plt.plot(frequencies[:max_freq_index], np.abs(fft_result[:max_freq_index]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.show()

frequencies = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

duration = 1.0

waves = [generate_sine_wave(duration, freq) for freq in frequencies]

create_wav_file("output.wav", waves)
plot_fft("output.wav", max_frequency=1000)
