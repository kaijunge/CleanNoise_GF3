from audioFunctions import *
from binaryFunctions import *
from qam import *
from sequenceGenerators import *
from receiver import *
from error_coding import *
from to_import import *



print("done importing")

CE_binary = binaryTextFileToBinary('Data_Files/random_bits.txt')
#binary = fileToBinary('Data_Files/kokoro_text.txt')
binary = binaryTextFileToBinary('Data_Files/kokoro_bin.txt')

print("coding binary")
CE_symbols = qpsk(CE_binary)
coded_binary = Hamming74(binary)


# Set the parameters for transmission
CP = 704 
N = 4096
guard = 5
CE_repeat = 20

frame_data_length = 180

# Make your instructions
instruction = []
for i in range(2047):
    if i <1500:
        instruction.append(1)
    else: 
        instruction.append(0)


symbols = varyingModulation_std(coded_binary,instruction, N, CE_binary)