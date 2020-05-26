from audioFunctions import *
from qam import *
from to_import import *

'''repeat = 20
timeshift = 676
known_data = F[5]
'''
N = 1024
K = 500
# y = recorded time series
# h = chirp signal (for matched filter)
# pause = pause time in seconds between chirp and data
def removeChirpAndPause(y, h, pause, plot = True, rng = 10):
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

def sliceDataContent(n, time_data_content, timeshift, N = N, K = K):
    dft = N + K
    samples_content = []
    freq_content = []
    # get the FFT of the data
    for i in range(repeat):
        samples_content.append(time_data_content[n*dft*repeat + dft*i:n*dft*repeat +dft*(i+1)][timeshift:timeshift+N])
        samples_content[i] = np.reshape(samples_content[i],np.zeros(N).shape)
        freq_content.append(fft(samples_content[i]))
    return samples_content, freq_content

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

def getImpulseSimple(freq, known_freq, N, repeat):
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
