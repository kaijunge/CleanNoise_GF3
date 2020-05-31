from to_import import *

sd.default.samplerate = fs  # Sample rate
sd.default.channels = 1


# Recording sound and saving it as a wav file
def record_and_save(filename, seconds):
    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("writing")
    write(filename, fs, myrecording)  # Save as WAV file 
    print("done")
    return myrecording

# Recording sound 
def record(seconds):
    myrecording = sd.rec(int(seconds * sd.default.samplerate))
    print("recording")
    print(sd.default.device)
    sd.wait()  # Wait until recording is finished
    print("done")
    return myrecording

# Play note on computer which is possible using a Bluetooth speaker
def play_note(note):
    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)

    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    print("Playing note")
        
    # Wait for playback to finish before exiting
    play_obj.wait_done()
    print("done")

# Play the reocrded data
def play(data, fs = fs):
    sd.play(data, fs)
    print("playing")
    print(sd.default.device)
    print(data)
    sd.wait()  # Wait until file is done playing

# Play the reocrded data, except now from an external file
def playFile(filename):
    # Extract data and sampling rate from file
    data, fs = sf.read(filename, dtype='float32')  
    sd.play(data, fs)
    print("playing")
    print(sd.default.device)
    print(data)
    sd.wait()  # Wait until file is done playing
    print("done")

# Return the array of values representing the wav file
def audioDataFromFile(filename):
    data, fs = sf.read(filename, dtype='float32')  
    return data