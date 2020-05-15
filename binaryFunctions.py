def toBinary(string):
    # Text to Binary
    return '0' + bin(int.from_bytes(string.encode(), 'big'))[2:]

def fileToBinary(filename):
    file = open(filename)
    line = file.read().replace("\n", " ")
    file.close()
    binary = '0' + bin(int.from_bytes(line.encode(), 'big'))[2:]

    return binary


    
def toText(binary_data):
    # Binary to Text
    n = int(binary_data, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode()

