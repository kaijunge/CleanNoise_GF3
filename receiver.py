from audioFunctions import *
from qam import *
from to_import import *

# y = recorded time series
# h = chirp signal (for matched filter)
# pause = pause time in seconds between chirp and data
def removeChirpAndPause(y, h, pause=0, plot = True, rng = 10):
    y = np.reshape(y, y.size)
    g = np.convolve(y, h[::-1], 'valid') # convoluton
    i_max = np.argmax(g[:int(len(g)/2)])
    time_start_index = int(i_max + len(h) + fs*pause) # Chirp ends here
    time_data = y[time_start_index:] # remove the chirp

    if plot: 
        plot_y(y, f=0, title="Received signal")
        plot_y(g, f=1, title="Match filter convolution")
        plot_y(g[i_max - rng:i_max + rng], f=2, title="Closeup of peak +- " + str(rng) + " samples")

    return time_data


def removeChirpAndPause_std(y, h, limit, plot = True, rng = 10):
    y_examin = np.array(y[:limit])
    g = np.convolve(y_examin, h[::-1], 'valid') # convoluton
    i_max = np.argmax(g[:int(len(g)/2)])
    time_start_index = int(i_max + len(h)) # Chirp ends here
    time_data = y[time_start_index:] # remove the chirp

    if plot: 
        plot_y(y_examin, f=0, title="Received signal")
        plot_y(g, f=1, title="Match filter convolution")
        plot_y(g[i_max - rng:i_max + rng], f=2, title="Closeup of peak +- " + str(rng) + " samples")

    return time_data



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

# Get the TF of the channel from a single OFDM symbol 
def getTF_SingleSymbol(symbol, known_freq, N, timeshift):    
    TF = np.zeros(N,dtype=complex)
    freq = fft(symbol[timeshift:timeshift + N])

    for i in range(N):
        if i == int(N/2) or i == 0:
            TF[i] == 0
        else:
            div = (freq[i]/known_freq[i] )
            TF[i] += div 

    impulse = ifft(TF)
    
    return impulse, TF


def sliceDataContent(data, timeshift, N_data, K_data, N_CE, K_CE, pilot_freq, total_data_symbol):
    dft = N + K


    samples_content = []
    freq_content = []
    # get the FFT of the data
    for i in range(repeat):
        samples_content.append(time_data_content[n*dft*repeat + dft*i:n*dft*repeat +dft*(i+1)][timeshift:timeshift+N])
        samples_content[i] = np.reshape(samples_content[i],np.zeros(N).shape)
        freq_content.append(fft(samples_content[i]))
    return samples_content, freq_content

def sliceDataContent_std(TF_front, TF_end, data, timeshift, N, CP, num_data_symbol, gradient):
    symbol_len = N + CP

    ### MAYBE DO SOMETHING LIKE COMPARE THE PHASE OF THIS ONE TO THE INITIAL ONE BUT RIGHT NOW IT'S FINE :P

    received_modulated_data = []
    # for every block of data we have
    for i in range(num_data_symbol):
    
        ## Maybe prepare some stuff for linear phase compensation.. maybe :P
        adjustment = gradient * -1 / (frame_data_length + CE_repeat)
            
        samples_content = np.array(data[ symbol_len*i : symbol_len*(i+1) ][timeshift:timeshift+N]) 
        freq_content = fft(samples_content)


        response = np.zeros(int((N/2) - 1), dtype = complex)

        for j in range(1,int(len(freq_content)/2)):
            div = (freq_content[j]/TF_front[j]) \
                * cmath.rect(1,j * int(i+ CE_repeat/2 + 0.5) * adjustment) 
            
            response[j-1] += div

        received_modulated_data.append(np.array(response))

            
    return received_modulated_data


def decode(maximum_freq_index, repetition_length, responses, num_info_blocks):
    # Decoding
    cutoff = maximum_freq_index-maximum_freq_index%repetition_length*8 + repetition_length*8

    recovery = []
    for n in range(num_info_blocks):
        
        relevant_data = responses[n][:cutoff]

        recovered_symbols = []
        for i in range(int(len(relevant_data)/repetition_length)):
            a = relevant_data[i*repetition_length:(i+1)*repetition_length]
            sequence = []
            for j in range(repetition_length):
                sequence.append(iqpsk(a[j]))

            recovered_symbols.append(stats.mode(sequence)[0])

        recovery.append(recovered_symbols)

    return recovery
    

def checkRecovery(maximum_freq_index, repetition_length, responses, num_info_blocks, relevant_data, known_data):
            
    for n in range(num_info_blocks):
        # Check with transmited symbols:
        transmitted_symbols = []
        F_ref = known_data[n][1:]
        for i in range(int(len(relevant_data)/repetition_length)):

            a = F_ref[i*repetition_length:(i+1)*repetition_length]

            sequence = []
            for j in range(repetition_length):
                sequence.append(iqpsk(a[j]))

            transmitted_symbols.append(stats.mode(sequence)[0])


        match_count = [0,0,0]

        # Get recovered symbols
        recovered_symbols = decode(maximum_freq_index, repetition_length, responses, num_info_blocks)

        for i in range(len(recovered_symbols)):
            #print(recovered_symbols[i], transmitted_symbols[i])
            if recovered_symbols[i] == transmitted_symbols[i]:
                #print("Match!")
                match_count[0] += 1
                match_count[2] += 1
            else:
                #print("no match....")
                match_count[1] += 1
                match_count[2] += 1

        print("Percent successful = ", 100*match_count[0]/match_count[2])



########################################################################
####   NOT USING THESE NOW   ###########################################
########################################################################

# Not using in main program - maybe in one of the week 2 challenge
# Get impulse response using a time average
def getImpulseSimple_time_avg(time_series, known_freq, N, repeat):
    TF = np.zeros(N,dtype=complex)
    for i, time in enumerate(time_series):
        if i == 0:
            total_time = time
        else:
            for j in range(len(time)):
                total_time[j] += time[j]
    
    avg_time = [x/repeat for x in total_time]
    freq_response = fft(avg_time)
    for i in range(N):
        if i == int(N/2) or i == 0:
            TF[i] == 0
        else:
            div = (freq_response[i]/known_freq[i] )
            TF[i] = div 
                
    impulse = ifft(TF)
    
    return impulse, TF


def getPhase(freq, repeat):
    Phase = np.zeros(511)
    for freq_response in freq:
        for i in range(1,int(len(freq_response)/2)):
            div = (freq_response[i]/known_data[i] )
            Phase[i-1] += cmath.phase(div)        
    Phase = [x/repeat for x in Phase]

    return Phase

def getPhase2(TF):
    Phase = [0]
    for i in range(1, int( len(TF)/2 )-1):
        Phase.append(cmath.phase(  TF[i]  ))
    
    return np.asarray(Phase)

# This is not working well, better solution needed
def getConvolutionMaximum(Phase):
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

    real_imax = i_max + (count_up - count_down)/2

    return real_imax

def getImpulse(freq, real_imax):
    TF = np.zeros(511,dtype=complex)
    TF_without_rotation = np.zeros(511,dtype=complex)
    for freq_response in freq:
        
        for i in range(1,int(len(freq_response)/2)):
            div = (freq_response[i]/known_data[i] ) * cmath.rect(1, math.pi* (0.01+(real_imax-20)*0.001) * i)
            div2 = (freq_response[i]/known_data[i] )
            TF[i-1] += div 
            TF_without_rotation[i-1] += div2
            
    TF = [x/repeat for x in TF]
    TF_without_rotation = [x/repeat for x in TF_without_rotation]
    impulse = ifft(TF)

    return impulse, TF_without_rotation

def getImpulse2(freq, real_imax, N):
    TF = np.zeros(1024,dtype=complex)
    TF_without_rotation = np.zeros(1024,dtype=complex)
    for freq_response in freq:
        
        for i in range(1024):
            if i == 512 or i == 0:
                TF[i] == 0
                TF_without_rotation[i] == 0
            else:
                div = (freq_response[i]/known[i] ) * cmath.rect(1, math.pi* (0.01+(real_imax-20)*0.001) * i)
                div2 = (freq_response[i]/known[i] )
                TF[i] += div 
                TF_without_rotation[i] += div2
                
    TF = [x/repeat for x in TF]
    TF_without_rotation = [x/repeat for x in TF_without_rotation]
    impulse = ifft(TF)
    
    return impulse, TF_without_rotation


def getResponse(freq_content, TF, real_imax, repeat):
    response = np.zeros(511, dtype = complex)
    for freq_response in freq_content:

        for i in range(1,int(len(freq_response)/2)):
            div = (freq_response[i]/ TF[i-1]) / cmath.rect(1, math.pi* (0.01+(real_imax)*0.001) * i)
            div2 = (freq_response[i]/TF[i-1] )

            response[i-1] += div2

    response = np.asarray([x/repeat for x in response])
    
    return response


def receive(h, N, K, repeat, timeshift, known_data):
    y, fs = sf.read('recording.wav') 
    # Remove chirp
    time_data = removeChirp(y, h)
    # Get samples and frequency arrays
    samples, freq = sliceData(time_data, timeshift)
    # Get phases of elements in freq array 
    Phase = getPhase(freq)

    real_imax = getConvolutionMaximum(Phase)

    impulse, TF_without_rotation = getImpulse(freq, real_imax)

    return impulse, TF_without_rotation


def receive2(y, time_start_index, repeat, N, K, num_info_blocks):

    y, fs = sf.read('recording.wav') 
    # Remove chirp
    time_data = removeChirp(y, h)
    # Get samples and frequency arrays
    samples, freq = sliceData(time_data, timeshift)
    # Get phases of elements in freq array 
    Phase = getPhase(freq)

    real_imax = getConvolutionMaximum(Phase)

    impulse, TF_without_rotation = getImpulse(freq, real_imax)

    time_data_content = removeChirpAndKnownData(y, h)

    # how many info blocks you sent
    responses = []
    for n in range(num_info_blocks):
        
        samples_content, freq_content = sliceDataContent(n, time_data_content, timeshift)

        response = getResponse(freq_content, TF_without_rotation, real_imax)

        responses.append(response)

    return responses


# slice data which has CE block for every n symbols
def sliceDataContent2(TF, data, timeshift_data, timeshift_CE, N_data, K_data, N_CE, K_CE, pilot_symbol_freq, total_data_symbol):
    data_length = N_data + K_data
    CE_length = N_CE + K_CE

    # block of PL1, PL2, ... , PLx , CE
    data_block_len = data_length*(pilot_symbol_freq - 1) + CE_length
    
    # number of these COMPLETE blocks in the data sequence
    num_data_block = math.floor(total_data_symbol/(pilot_symbol_freq-1)) 
    remaining_symbols = total_data_symbol - num_data_block * (pilot_symbol_freq-1)
    
    received_modulated_data = []
    # for every block of data we have
    for i in range(num_data_block + 1):
        
        # For any remainders
        if i == num_data_block and remaining_symbols != 0:
    
            remainder_block_len = data_length * remaining_symbols
            block = data[i*data_block_len:i*data_block_len + remainder_block_len + CE_length]
            
            impulse_local, TF_local = getTF_SingleSymbol(block[-1*CE_length:], known_freq, CE_N, timeshift_CE)
            
            ### MAYBE DO SOMETHING LIKE COMPARE THE PHASE OF THIS ONE TO THE INITIAL ONE BUT RIGHT NOW IT'S FINE :P

            symbols_in_block = remaining_symbols
            
        # normal procedure
        else:
            
            # get the particular data block in question
            block = data[i*data_block_len:(i+1)*data_block_len]

            # find the TF in this block
            impulse_local, TF_local = getTF_SingleSymbol(np.roll(block[-1*CE_length:],0), known_freq, CE_N, timeshift_CE)

            ### MAYBE DO SOMETHING LIKE COMPARE THE PHASE OF THIS ONE TO THE INITIAL ONE BUT RIGHT NOW IT'S FINE :P

            symbols_in_block = pilot_symbol_freq-1
            
        
        start_index = 200
        end_index = 200 + 512
        TF_local_roll = fft(np.roll(impulse_local, -1*guard-3))[start_index:end_index]
        TF_roll = fft(np.roll(impulse_simple, -1*guard-3))[start_index:end_index]
        
        angle_local = np.angle(TF_local_roll)
        angle_reference = np.angle(TF_roll)
        
        difference = []
        for p, val in enumerate(angle_local):
            difference.append(val - angle_reference[p])
        
        x = np.linspace(start_index, end_index, end_index- start_index)
        param = np.polyfit(x, difference, 1)
        
        plot_y(difference, f = i+100, title = i)
        print(param, i)
        
        #plot_y(angle_local, f = i, title = i)
        #plot_y(angle_reference, f= i)
        # for every Payload symbol in this data block 
        for j in range(symbols_in_block):
            
            samples_content = np.array(block[ data_length*j : data_length*(j+1) ][timeshift_data:timeshift_data+N_data]) 
            freq_content = fft(samples_content)
            

            response = np.zeros(int((Payload_N/2) - 1), dtype = complex)

            for k in range(1,int(len(freq_content)/2)):
                div = (freq_content[k]/TF[k]) * cmath.rect(1,2*math.pi * k * ((i+1)*0.15+(j+1)*0.05)) # <- Add phase correction
                response[k-1] += div
        
            received_modulated_data.append(np.array(response))
            
            
    return received_modulated_data

