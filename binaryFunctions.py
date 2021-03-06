from to_import import *

# convert a string into binary using utf-8 encoding
def toBinary(string):
    # Text to Binary
    return '0' + bin(int.from_bytes(string.encode(), 'big'))[2:]

# convert a text file into a string of binary 
def fileToBinary(filename):
    file = open(filename)
    line = file.read().replace("\n", " ")
    file.close()
    binary = '0' + bin(int.from_bytes(line.encode(), 'big'))[2:]
    return binary
    
# binary to text but it doesn't work sometimes so using str_to_bytearray instead
def toText(binary_data):
    # Binary to Text
    n = int(binary_data, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

# return a string of binary from a text file of binary information
def binaryTextFileToBinary(filename):
    file = open(filename)
    line = file.read().replace("\n", " ")
    file.close()
    return line

# Binary string to Bytes
def str_to_bytearray(string_data):
    new_data = []
    for i in range(0, len(string_data), 8):
        new_data.append(string_data[i:i+8])  

    int_data = [] 
    for i in new_data:
        int_data.append(int(i,2))

    return bytearray(int_data)

# XOR function
_xormap = {('0', '1'): '1', ('1', '0'): '1', ('1', '1'): '0', ('0', '0'): '0'}
def xor(x, y):
    return ''.join([_xormap[a, b] for a, b in zip(x, y)])

# Reutrn a binary file from some file (some file -> byte -> binary)
def imageFileToBinary(filename):
    with open(filename, "rb") as image:
        f = image.read()
        b = bytearray(f)

    return '0' + bin(int.from_bytes(b, 'big'))[2:]