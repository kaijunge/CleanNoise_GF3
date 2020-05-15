import numpy as np
import scipy.signal
from variables import *
from qam import *
from audioFunctions import *
import cmath
from scipy.fftpack import fft, fftshift, ifft, fftfreq

def scaleToAudible(array):
    scale_factor = 2**15 - 1
    scaled_array = np.int16(array/np.max(np.abs(array)) * scale_factor) #Scaling

    return scaled_array

#Generate a Zadoff-Chu sequence of given order and length
def ZadoffChu(order, length, index=0):
    cf = length % 2
    n = np.arange(length)
    arg = np.pi * order * n * (n+cf+2*index)/length
    return np.exp(-1j*arg)

#Generate a frequency sweep sequence in a given range for a given duration
def Chirp(f0, f1, t1):
    t = np.linspace(0, t1, t1 * fs, False)
    chirp = scipy.signal.chirp(t, f0, t1, f1)
    chirp = scaleToAudible(chirp) #Scaling

    return chirp

#Generate a sequence of zeros for a given duration
def Pause(seconds):
    return np.zeros(fs*seconds)


#Create repetitions of symbols
def repetitionCoding(symbol_array, num_of_repeats):

    repetition_code = []

    for symbol in symbol_array:
        for i in range(num_of_repeats):
            repetition_code.append(symbol)

    return repetition_code

#Assign encoded_symbols to OFDM symbols
def ofdmSymbols(encoded_symbols, CP_length, DFT_length, max_freq_index):
    assert DFT_length>=CP_length, "CP length must be <= DFT length"

    info_block_length = int(DFT_length/2-1)
    ofdm_symbol_array = []
    index = 0
    limit = min(max_freq_index, len(encoded_symbols))
    print(limit)
    while index < len(encoded_symbols):
        info_block = []
        for i in range(limit): #info_block_length
            print(i)
            info_block.append(encoded_symbols[i])
            index += 1
        print(len(info_block))
        print(info_block_length)
        padding = info_block_length - len(info_block)
        #print(len(chunk), "blah")
        for i in range(padding):
            # not sure what to add on these blocks... update: works very well removing high freq stuff
            info_block.append(cmath.rect(0.1,0))#cmath.rect(1, math.pi/4))
        info_block = np.asarray(info_block)
        info_block_reverse_conjugate = info_block[::-1].conjugate()
        useful_data_frequencies = np.concatenate(([0],info_block, [0],info_block_reverse_conjugate))
        
        useful_data = ifft(useful_data_frequencies).real
        ofdm_symbol = np.concatenate((useful_data[-1*CP_length:], useful_data))
        ofdm_symbol = scaleToAudible(ofdm_symbol)
        ofdm_symbol_array.extend(ofdm_symbol)
        
    return ofdm_symbol_array, useful_data_frequencies

def transmit(ofdm_symbol_array):
    
    output = np.concatenate((Chirp(2000, 4000, 1),Pause(1),ofdm_symbol_array,Pause(1)))
    write('chirp_signal_4.wav', 44100, output)

    play_np_BT(output)

