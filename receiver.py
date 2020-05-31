from audioFunctions import *
from qam import *
from to_import import *
from binaryFunctions import *

# y = recorded time series
# h = chirp signal (for matched filter)
# pause = pause time in seconds between chirp and data
def removeChirpAndPause_std(y, h, limit, plot = True, rng = 10):
    y_examin = np.array(y[:limit])
    g = np.convolve(y_examin, h[::-1], 'valid') # convoluton
    i_max = np.argmax(g[:int(len(g)/2)])
    time_start_index = int(i_max + len(h)) # Chirp ends here
    time_data = y[time_start_index:] # remove the chirp

    if plot: 
        print("i_max", i_max)
        plot_y(y_examin, f=0, title="Received signal")
        plot_y(g, f=1, title="Match filter convolution")
        plot_y(g[i_max - rng:i_max + rng], f=2, title="Closeup of peak +- " + str(rng) + " samples")

    return time_data


def detect_chirps(y, h, limit):
    
    g = np.correlate(y[:limit*fs], h, 'valid')
    half_max = max(g)/2
    
    near_max = 0
    for i, val in enumerate(g):
        if val > half_max:
            near_max = i
            break
            
    if near_max - fs*2 >= 0:
        return y[near_max - fs*2:]
    else:
        return y


def count_frames(y_next, h, length):
    i = 0
    count = 1
    while 1:
        portion = y_next[:fs*5]
        g = np.convolve(portion, h[::-1], 'valid')
        max_value = max(g)

        if i == 0:
            compare = max_value
        else:
            if max_value > compare * 0.2: 
                count += 1
            else:
                break

        y_next = y_next[length:]
        if len(y_next) < len(h)*2:
            break

        i += 1

    return count



# return the array of relevant N time domain samples from an obtained time series
# Input:    time_data = time series which we assume the data starts at n=0
#           timeshift = where we want to start collecting the data (some point in the CP)
#           N         = OFDM length
#           K         = CP length
#           repeat    = Number of times each CP+OFDM is repeated
# Output:   samples   = np array of time series which should be circular_conc(x, h)
#           freq      = np array of freq series which should be circular_conc(x, h)
#           remaining = what is left of the original time series after cutting off the data
def sliceData(time_data, timeshift, N, K, repeat):
    dft = N + K
    samples = []
    freq = []
    for i in range(repeat):
        samples.append(time_data[dft*i:dft*(i+1)][timeshift:timeshift+N])
        samples[i] = np.reshape(samples[i],np.zeros(N).shape)
        freq.append(fft(samples[i]))

    remaining = time_data[dft*repeat:]

    return np.asarray(samples), np.asarray(freq), remaining

# Get the TF of the channel by averaging multiple OFDM symbols 
def getTF_FreqAverage(freq, known_freq, N, repeat):
    TF = np.zeros(N,dtype=complex)
    for freq_response in freq:
        
        for i in range(N):
            if i == int(N/2) or i == 0:
                TF[i] == 0
            else:
                div = (freq_response[i]/known_freq[i] )
                TF[i] += div 
                
    TF = [x/repeat for x in TF]
    impulse = ifft(TF)
    
    return impulse, TF


def sliceDataContent_std(TF_front, TF_end, data, timeshift, N, CP, num_data_symbol, gradient, CE_repeat):
    symbol_len = N + CP

    ### MAYBE DO SOMETHING LIKE COMPARE THE PHASE OF THIS ONE TO THE INITIAL ONE BUT RIGHT NOW IT'S FINE :P

    received_modulated_data = []
    # for every block of data we have
    for i in range(num_data_symbol):
    
        ## Maybe prepare some stuff for linear phase compensation.. maybe :P
        adjustment = gradient * -1 / (num_data_symbol + CE_repeat)
            
        samples_content = np.array(data[ symbol_len*i : symbol_len*(i+1) ][timeshift:timeshift+N]) 
        freq_content = fft(samples_content)


        response = np.zeros(int((N/2) - 1), dtype = complex)

        for j in range(1,int(len(freq_content)/2)):
            div = (freq_content[j]/TF_front[j]) \
                * cmath.rect(1,j * int(i+ CE_repeat/2 + 0.5) * adjustment) 
            
            response[j-1] += div

        received_modulated_data.append(np.array(response))

            
    return received_modulated_data

def demodVaryingModulation_std(constellation_array, instruct_list, N):
    inst_len = int(N/2 -1)
    assert inst_len == len(instruct_list), "instruction list length must match DFT length"
  
    binary_block = []
    
    
    i = 0 #variable to iterate through the instructions symbols
    
    for j in range(len(constellation_array)):
            
        if instruct_list[i] == 1:
            binary_block.append(iqpsk(constellation_array[j:j+1]))
        
        elif instruct_list[i] == 2: 
            binary_block.append(iqam16(constellation_array[j:j+1]))
        
        elif instruct_list[i] == 3: 
            binary_block.append(ibpsk(constellation_array[j:j+1]))
        else:
            pass
        
        i += 1
        
        if i > inst_len-1:
            i = 0
    
    return "".join(binary_block) 

# Find information about the file name and the byte length of the file
def remove_metadata(binary_recovered):
    file_seperation = []
    count = 0
    for i in range(1000):
        byte = binary_recovered[i*8:(i+1)*8]
        zero = True
        for bit in byte:
            if bit == '1':
                zero = False

        if zero:
            file_seperation.append(i*8)

            count += 1

        if count == 2:
            break

    # Extract file name and length of the file
    Filename= str_to_bytearray(binary_recovered[0:file_seperation[0]]).decode("utf-8", "replace")
    Length =  int(str_to_bytearray(binary_recovered[file_seperation[0]+8:file_seperation[1]]).decode("utf-8", "replace"))

    extra_bits = len(binary_recovered) - file_seperation[1] - 8 - Length*8

    raw_file = binary_recovered[file_seperation[1] + 8 : -1*extra_bits]
    
    return Filename, Length, raw_file

# K-Means
def find_angle_offset(a, disp = False):

    b = [[None]] * len(a)
    mag = []
    for i in range(len(a)):
        mag.append(abs(a[i]))
        if abs(a[i]) < 5:
            b[i] = [a[i].real, a[i].imag]
        else:
            b[i] = [a[i].real/abs(a[i]), a[i].imag/abs(a[i])]


    kmeans = KMeans(n_clusters=4, random_state=0).fit(b)

    cent = kmeans.cluster_centers_

    #x,y = np.split(cent, 2,axis = 1)
    #plt.scatter(x, y)


    angle = []
    for i in range(4):
        angle.append(np.angle(cent[i][0] + cent[i][1]*1j))
    
    angle = np.array(angle)

    
    ## this depends on the data but for now we penalise everything with greater abs phase than pi/4
    angle2 = angle
    for i in range(len(angle2)):
        if abs(angle2[i]) > math.pi*(7/18):
            angle2[i] += 100
    
    #c = abs(angle - np.ones(angle.shape)*(math.pi/4))
    c = abs(angle2 - np.ones(angle.shape)*(math.pi/4))
    
    output =  math.pi/4 - angle2[c.argmin()]
    
    if disp:
        print(angle, c, c.argmin(), output,end = '\n\n')
        print(cent)
        #print(angle[np.abs(angle - math.pi/4).argmin()])
        #print("closest to pi/4", angle[c.argmin()])

    return output 
