import numpy as np
import cmath, math
from scipy.fftpack import fft, fftshift, ifft, fftfreq
from scipy import stats
from scipy.signal import chirp, correlate, spectrogram
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import simpleaudio as sa
import wave
import scipy
import random
from sklearn.cluster import KMeans

# Sampling frequency
fs = 48000 #44100

# Function to display something quickly
def plot_y(y, f=0, title = ""):
    plt.figure(f)
    x = np.linspace(0, len(y), len(y))
    plt.plot(x, y)
    if title != "":
        plt.title(title)
    plt.show