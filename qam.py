from to_import import *

# Take a sequence of bits and returns a sequence of BPSK symbols half the length 
def bpsk(binary):
    #bits_per_symbol = 1
    
    symbols = []
    for i in range(len(binary)):
        real = 1
        imaginary = 0
        
        if binary[i] == '0': 
            real *= -1
        symbols.append(real + imaginary)
        
    return np.asarray(symbols)


# given a complex number constellation return the constellation (0,1)
def ibpsk(complex_number_array):
    binary = ""
    for complex_number in complex_number_array:
        Re = complex_number.real
        # decision regions
        bit = "1"
        if Re < 0:
            bit = "0"

        binary += bit
        

    return binary

# Take a sequence of bits and returns a sequence of QPSK symbols 
def qpsk(binary):
    assert len(binary)%2 == 0, "Binary string should have length multiple of 2"
    
    #bits_per_symbol = 2
    
    symbols = []
    for i in range(int(len(binary)/2)):
        real = 1/math.sqrt(2)
        imaginary = 1/math.sqrt(2)*1j
        
        index = i*2 #every 2 bits
        if binary[index] == '1': 
            imaginary *= -1
        
        if binary[index+1] == '1': 
            real *= -1
            
        symbols.append(real + imaginary)
        
    return np.asarray(symbols)

# given QPSK symbols return a sequence of bits
def iqpsk(complex_number_array):
    binary = ""
    for complex_number in complex_number_array:
        Re = complex_number.real
        Im = complex_number.imag
        # decision regions
        bit1 = "0"
        bit2 = "0"
        if Im < 0:
            bit1 = "1"
        if Re < 0:
            bit2 = "1"
        
        binary += bit1 + bit2
        
    return binary

# Take a sequence of bits and returns a sequence of 16QAM symbols half the length 
def qam16(binary):
    assert len(binary)%4 == 0, "Binary string should have length multiple of 4"
    
    #bits_per_symbol = 4
    
    symbols = []
    
    modulation = {
    '0000' : (0.25, 0.25),
    '0001' : (0.75, 0.25),
    '0010' : (0.25, 0.75),
    '0011' : (0.75, 0.75),
    '0100' : (0.25, -0.25),
    '0101' : (0.75, -0.25),
    '0110' : (0.25, -0.75),
    '0111' : (0.75, -0.75),
    '1000' : (-0.25, 0.25),
    '1001' : (-0.75, 0.25),
    '1010' : (-0.25, 0.75),
    '1011' : (-0.75,  0.75),
    '1100' : (-0.25, -0.25),
    '1101' : (-0.75, -0.25),
    '1110' : (-0.25, -0.75),
    '1111' : (-0.75, -0.75),
    }
    
    for i in range(int(len(binary)/4)):
        index = i*4
        name = binary[index:index+4]
        real = modulation[str(name)][0]
        imaginary = modulation[str(name)][1] * 1j
            
        symbols.append(real + imaginary)
        
    return np.asarray(symbols)

def iqam16(complex_number_array):
    binary = ""
    for complex_number in complex_number_array:
        Re = complex_number.real
        Im = complex_number.imag
        # decision regions
        bit1 = "0"
        bit2 = "0"
        bit3 = "0"
        bit4 = "0"
        if Re < 0:
            bit1 = "1"
        if Im < 0:
            bit2 = "1"
        if abs(Im) > 0.5:
            bit3 = "1"
        if abs(Re) > 0.5:
            bit4 = "1"
        
        binary += bit1 + bit2 + bit3 + bit4
    
    return binary

#Returns array of constellation symbols where the user can specify the modulation sheme for each packet of bits
#data should be a string of binary digits eg "00101010010101001001"
#instruct_str - eg [0,1,1,2] where 0 corresponds to zero-padding, 1 - QPSK, 2 - 16QAM

def varyingModulation(data, instruct_str):
    
    info_block = instruct_str
    
    j = 0 #variable to iterate through the binary data
    i = 0 #variable to iterate through the instructions 
    
    while j < len(data):
        if instruct_str[i] == 0:
            info_block[i] = 0
            j += 1
            i += 1
            
        elif instruct_str[i] == 1:
            info_block[i] = qpsk(data[j:j+2])
            j += 2
            i += 1
        
        elif instruct_str[i] == 2: 
            info_block[i] = qam16(data[j:j+4])
            j += 4
            i += 1
            
    info_block = np.asarray(info_block)
    
    info_block_reverse_conjugate = info_block[::-1].conjugate()
    
    info_block, info_block_reverse_conjugate = info_block.ravel(), info_block_reverse_conjugate.ravel() 
    
    useful_data_frequencies = np.concatenate(([0],info_block, [0],info_block_reverse_conjugate))
    
    return useful_data_frequencies



def varyingModulation_std(data, instruct_list, N, random_bits, max_odfm_symbol):
    
    inst_len = int(N/2 -1)
    assert inst_len == len(instruct_list), "instruction list length must match DFT length"
  
    bitcount = 0
    for instruction in instruct_list:
        bitcount += instruction*2
        
    
    additional = bitcount -len(data)%bitcount 
    new_data = data + random_bits[:additional]
    
    print("bitcount", bitcount)
    print("additional ", additional)
    print("data length ", len(data))
    print("new data len ", len(new_data))
    
    symbol_length = int((len(new_data)/bitcount) * len(instruct_list))
    info_block = np.zeros(symbol_length,dtype=complex)
    
    j = 0 #variable to iterate through the binary data
    i = 0 #variable to iterate through the instructions 
    k = 0 #variable to iterate through the output data
    
    ofdm_symbol_count = 0
    while j < len(new_data):
        if instruct_list[i] == 0:
            info_block[k] = complex(qpsk(random_bits[i:i+2]))
            j += 0

        elif instruct_list[i] == 1:
            info_block[k] = complex(qpsk(new_data[j:j+2]))
            j += 2
        
        elif instruct_list[i] == 2: 
            info_block[k] = complex(qam16(new_data[j:j+4]))
            j += 4
            
        i+= 1
        k+= 1
        
        if i == inst_len:
            i = 0
            ofdm_symbol_count += 1

            if ofdm_symbol_count%100 == 0:
                print("symbol number " + str(ofdm_symbol_count) + " is done.")

            if ofdm_symbol_count == max_odfm_symbol:
                break
            
    info_block = np.array(info_block).ravel()
    return info_block