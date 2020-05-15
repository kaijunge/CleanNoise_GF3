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
    while index < len(encoded_symbols):
        info_block = []
        for i in range(limit): #info_block_length
            info_block.append(encoded_symbols[i])
            index += 1
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

def transmit2(*symbol_array):
    output = Pause(1)
    for symbol in symbol_array:
        output = np.append(output, symbol) 

    play_note(output)




'''
h = Chirp(2000, 4000, 1)
timeshift = 676



def receive(known_symbols, N, K, h, timeshift):
    
    # record for some seconds
    y = record(12)
    data = np.reshape(y, y.size)
    # Chirp for matched filter NB should be same as the transmitted chirp!
    h_rev = h[::-1]
    g = np.convolve(data, h_rev, 'valid') #convolve
    centre = int(i_max = np.argmax(g)) #find maximum
    dft = N + K 
    time_start_index = centre + len(h_rev) + 44100 + dft*0
    time_data = y[time_start_index:]
    dft = N + K 

    samples = []
    freq = []

    for i in range(repeat):
        samples.append(time_data[dft*i:dft*(i+1)][timeshift:timeshift+1024])
        samples[i] = np.reshape(samples[i],np.zeros(1024).shape)
        freq.append(fft(samples[i]))
        
    known = F[5]
    Phase = np.zeros(511)
    for freq_response in freq:
        
        for i in range(1,int(len(freq_response)/2)):
            div = (freq_response[i]/known[i] )
            Phase[i-1] += cmath.phase(div)        
            
    Phase = [x/repeat for x in Phase]

    max_conv = []
    for s in range(1000):
        P = []
        coeff = -0.01-s*0.001
        n = 0
        sign = coeff/abs(coeff)
        for i in range(1,511):
            val = math.pi * coeff * i - sign *2*math.pi*n

            if sign > 0:
                if val > math.pi:
                    val = val - 2*math.pi
                    n += 1
            else:
                if val < -1*math.pi:
                    val = val + 2*math.pi
                    n += 1   
                    
            P.append(val)
            
        max_conv.append( max(correlate(P, Phase)))

        max_conv = np.asarray(max_conv)
        i_max = np.argmax(max_conv)
        max_conv_max = np.max(max_conv)

        count_up = 0
        count_down = 0
        cutoff = 0.75
        for i in range(100):
            if max_conv[i_max+i] > max_conv_max*cutoff:
                count_up +=1
            
            if max_conv[i_max-i] > max_conv_max*cutoff:
                count_down +=1

        print(count_up, count_down)

        real_imax = i_max + (count_up - count_down)/2


        #plot_y(max_conv[600:700])
        print("max ", max_conv_max, "i_max ", i_max, "real i_max ", real_imax)


        TF = np.zeros(511,dtype=complex)
        TF_without_rotation = np.zeros(511,dtype=complex)
        for freq_response in freq:
            
            for i in range(1,int(len(freq_response)/2)):
                div = (freq_response[i]/known[i] ) * cmath.rect(1, math.pi* (0.01+(real_imax-20)*0.001) * i)
                div2 = (freq_response[i]/known[i] )
                TF[i-1] += div 
                TF_without_rotation[i-1] += div2
                
        TF = [x/repeat for x in TF]
        TF_without_rotation = [x/repeat for x in TF_without_rotation]

        impulse = ifft(TF)
        plot_y(impulse)
        '''