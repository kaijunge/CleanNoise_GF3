from qam import *
from audioFunctions import *
from to_import import *

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
    chirp_signal = chirp(t, f0, t1, f1)
    chirp_signal = scaleToAudible(chirp) #Scaling

    return chirp_signal

#Generate a sequence of zeros for a given duration
def Pause(seconds):
    return np.zeros(fs*seconds)


#Create repetitions of symbols
def repetitionCoding(symbol_array, num_of_repeats):

    repetition_code = []

    for symbol in symbol_array:
        for i in range(num_of_repeats):
            repetition_code.append(symbol)

    return np.asarray(repetition_code)

#Assign encoded_symbols to OFDM symbols
def ofdmSymbols(encoded_symbols, CP_length, DFT_length, max_freq_index):
    assert DFT_length>=CP_length, "CP length must be <= DFT length"

    info_block_length = int(DFT_length/2-1)
    limit = min(max_freq_index, info_block_length)

    # output list
    ofdm_time_arrays = []
    ofdm_long_time_array = []
    ofdm_freq_arrays = []
    index = 0
    # Loop through all the symbols
    while index < len(encoded_symbols):
        
        # Add symbols in one DFT block, based on the limit of
        info_block = []
        for _ in range(limit):
            try: 
                info_block.append(encoded_symbols[index])
            except:
                break
            index += 1

        # Add some information to pad the block of information so it is length = N/2 - 1
        padding = info_block_length - len(info_block)
        for _ in range(padding):
            # this is arbiturary, it can be something else
            info_block.append(cmath.rect(0.1,0))
        
        # Append complex conjugate and zeros to produce a real time series producing OFDM symbol sequence
        info_block = np.asarray(info_block)
        info_block_reverse_conjugate = info_block[::-1].conjugate()
        useful_data_frequencies = np.concatenate(([0],info_block, [0],info_block_reverse_conjugate))
        
        # iFFT to produce time domain signal
        useful_data = ifft(useful_data_frequencies).real

        # Add cyclic prefix
        if CP_length == 0:
            ofdm_single_time_domain = useful_data
        else:
            ofdm_single_time_domain = np.concatenate((useful_data[-1*CP_length:], useful_data))

        ofdm_single_time_domain = scaleToAudible(ofdm_single_time_domain)
        ofdm_long_time_array.extend(ofdm_single_time_domain)

        ofdm_freq_arrays.append(useful_data_frequencies)
        ofdm_time_arrays.append(ofdm_single_time_domain)
        
    return ofdm_time_arrays, ofdm_freq_arrays, ofdm_long_time_array

# repeat some signal n times (can input 1D or 2D array)
def repeat_signal(data, repeat_number):
    return np.tile(data, repeat_number)

def transmit2(*symbol_array):
    output = Pause(1)
    for symbol in symbol_array:
        output = np.append(output, symbol) 

    play_note(output)






def transmit(chirp_signal, ofdm_symbol_array, play = False):
    
    output = np.concatenate(chirp_signal,Pause(1),ofdm_symbol_array,Pause(1))
    write('chirp_signal_4.wav', 44100, output)

    if play:
        play_note(output)
