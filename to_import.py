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
import scipy


fs = 44100