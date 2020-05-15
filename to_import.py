import numpy as np
import cmath, math
from scipy.fftpack import fft, fftshift, ifft, fftfreq
from scipy import stats
from scipy.signal import chirp, correlate
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
import wave


fs = 44100

# Function to display something quickly
def plot_y(y, f=0):
    plt.figure(f)
    x = np.linspace(0, len(y), len(y))
    plt.plot(x, y)
    plt.show