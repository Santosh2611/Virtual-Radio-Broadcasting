import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
from scipy.io import wavfile as wav # Read data from and write data to file
from pydub import AudioSegment as AudioSegment # Import and manipulate audio files
import wave as wave # Provides a convenient interface to the WAV sound format

def Create_Music_From_Notes(): # Create Music Using Given Notes
    
    # Calculate note frequency:
    def get_piano_notes():   
        
        # White keys are in uppercase and black keys (sharps) are in lowercase:
        octave = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B'] 
        base_freq = 440 # Frequency of note A4, the 49th key
        keys = np.array([x+str(y) for y in range(0,9) for x in octave])
        
        # Trim to standard 88 keys:
        start = np.where(keys == 'A0')[0][0]
        end = np.where(keys == 'C8')[0][0]
        keys = keys[start:end+1]
        
        # Returns a dictionary that maps a note name to corresponding frequency in Hertz:
        note_freqs = dict(zip(keys, [2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
        note_freqs[''] = 0.0 # stop
        return note_freqs
    
    # Create a sine wave:
    def get_sine_wave(frequency, duration, sample_rate=44100, amplitude=4096):
        t = np.linspace(0, duration, int(sample_rate*duration)) # Define the Time Axis
        wave = amplitude*np.sin(2*np.pi*frequency*t)
        return wave
    
    # An audio file containing the selected note is created and stored:
    def createfile(x2,i):
         frequency = note_freqs[x2]
         sine_wave = get_sine_wave(frequency, duration=1, amplitude=2048);
         wav.write('Sound_' + str(i) + '.wav', rate=44100, data=sine_wave.astype(np.int16));
        
    # Combine the audio files:
    def combine_file():
    
        sum = AudioSegment.from_file("E:\Plan B\Amrita Vishwa Vidyapeetham\Subject Materials\Semester III\Signal Processing Lab (19CCE281)\Project\Sound_1.wav", format="wav") # Initialize first audio file
        
        # Add the successive files:
        for a in range (2,i):
            
            prev = AudioSegment.from_file("E:\Plan B\Amrita Vishwa Vidyapeetham\Subject Materials\Semester III\Signal Processing Lab (19CCE281)\Project\Sound_" + str(a) + ".wav", format="wav")
            
            sum = sum + prev
            
            sum.export("E:\Plan B\Amrita Vishwa Vidyapeetham\Subject Materials\Semester III\Signal Processing Lab (19CCE281)\Project\Output.wav", format="wav")
    
    # Get input from the user:
    
    i = 0
    
    print("\nMENU:\n   1. C\n   2. C#\n   3. D\n   4. D#\n   5. E\n   6. F\n   7. F#\n   8. G.\n   9. G#\n   10. A\n   11. A#\n   12. B\n   13. Exit\nEnter the number corresponding to the menu to implement the note one by one as follows: ")
    
    while True:
        
        i = i + 1        
        note_freqs = get_piano_notes() # Calculate note frequency
        note = int(input("Note " + str(i) + ": "))
        
        if note == 1:
            temp = 'C4'
            createfile(temp, i)
        elif note == 2:
            temp = 'c4'
            createfile(temp, i)
        elif note == 3:
            temp = 'D4'
            createfile(temp, i)
        elif note == 4:
            temp = 'd4'
            createfile(temp, i)
        elif note == 5:
            temp = 'E4'
            createfile(temp, i)
        elif note == 6:
            temp = 'F4'
            createfile(temp, i)
        elif note == 7:
            temp = 'f4'
            createfile(temp, i)
        elif note == 8:
            temp = 'G4'
            createfile(temp, i)
        elif note == 9:
            temp = 'g4'
            createfile(temp, i)
        elif note == 10:
            temp = 'A4'
            createfile(temp, i)
        elif note == 11:
            temp = 'a4'
            createfile(temp, i)
        elif note == 12:
            temp = 'B4'
            createfile(temp, i)
        elif note == 13:
            break
        else:
            print("Error: Invalid Input! Please try again.")
                   
    combine_file() # Combine the audio files
    
    # Read and Open the Generated Music File:
    rate, data = wav.read('Output.wav')  
    raw = wave.open('Output.wav')
    f_rate = raw.getframerate()
    time = np.linspace(
    0, # start
    len(data) / f_rate,
    num = len(data)
    )
    
    # Plot Generated Music Signal:
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.title("Generated Music Signal")
    plt.plot(time, data)
    plt.grid(True) # Configure the grid lines
    plt.show()
    
Create_Music_From_Notes() # Create Music Using Given Notes
