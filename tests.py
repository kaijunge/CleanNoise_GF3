from audioFunctions import *
from binaryFunctions import *
from qam import *
import unittest 
  
class SimpleTest(unittest.TestCase): 
  
    # Returns True or False.  
    def testBinaryConversion(self):        
        message = 'Hello World'
        binary = toBinary(message)
        recovered_message = toText(binary)
        self.assertTrue(message == recovered_message) 
    
    def testQPSK(self):        
        message = 'Hello World'
        binary = toBinary(message)
        symbols = qpsk(binary)
        recovered_binary = iqpsk(symbols)
        self.assertTrue(binary == recovered_binary) 

    def testBPSK(self):        
        message = 'Hello World'
        binary = toBinary(message)
        symbols = bpsk(binary)
        recovered_binary = ibpsk(symbols)
        self.assertTrue(binary == recovered_binary) 

    def testQAM16(self):        
        message = 'Hello World'
        binary = toBinary(message)
        symbols = qam16(binary)
        recovered_binary = iqam16(symbols)
        print(binary)
        print(symbols)
        print(recovered_binary)
        self.assertTrue(binary == recovered_binary) 

if __name__ == '__main__': 
    unittest.main() 