from audioFunctions import *
from binaryFunctions import *
from qam import *
from sequenceGenerators import *

message = 'Hello World'
print(message)

binary = toBinary(message)
print(binary)

symbols = qpsk(binary)

repeated_symbols = repetitionCoding(symbols, 2)
print('repeated_symbols: ' + str(len(repeated_symbols)))

ofdm_symbols, useful_data_frequencies = ofdmSymbols(repeated_symbols, 1024, 1024, 511)
print(ofdm_symbols)

plot_y(ofdm_symbols)