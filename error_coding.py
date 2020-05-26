from binaryFunctions import *
from to_import import *


#Create repetitions of symbols
def repetitionCoding(symbol_array, num_of_repeats):

    repetition_code = []

    for symbol in symbol_array:
        for i in range(num_of_repeats):
            repetition_code.append(symbol)

    return np.asarray(repetition_code)

    
#convert binary string to Hamming(7,4) code
def Hamming74(binary):
    
    y = binaryToArray(binary)
    
    G = np.array([
        [1,1,0,1],
        [1,0,1,1],
        [1,0,0,0],
        [0,1,1,1],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ])

    output = ''

    for i in range(len(y)//4):
        data_block = y[i*4:(i+1)*4]
        result = np.matmul(G, data_block)%2
        result = np.array2string(result).replace(" ", "").replace("[", "").replace("]", "")
        output += result
    return output

#convert Hamming(7,4) code to binary string
def iHamming74(binary):
    
    y = binaryToArray(binary)
    
    H = np.array([
        [1,0,1,0,1,0,1],
        [0,1,1,0,0,1,1],
        [0,0,0,1,1,1,1]
    ])
    
    R = np.array([
        [0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0],
        [0,0,0,0,0,1,0],
        [0,0,0,0,0,0,1]
    ])
    
    output = ''
    
    for i in range(len(y)//7):
        data_block = y[i*7:(i+1)*7]
        error_check = np.matmul(H, data_block)%2
        if (error_check != 0).any():
            error_index = error_check[0] + 2*error_check[1] + 4*error_check[2] - 1  
            data_block[error_index] = (data_block[error_index]+1)%2
        result = np.matmul(R, data_block)%2
        result = np.array2string(result).replace(" ", "").replace("[", "").replace("]", "")
        result = str(result)
        output += result
    return output