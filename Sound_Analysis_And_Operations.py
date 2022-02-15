import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
import sounddevice as sd # Play and record NumPy arrays containing audio signals
import soundfile as sn # Read and write sound files
import wave as wave # Provides a convenient interface to the WAV sound format
import math as math # Provides access to the mathematical functions
import struct as struct # Used in handling binary data stored in files
import random as random # Implements pseudo-random number generators
import scipy.io.wavfile as sc # Open a Waveform Audio File (WAV) Format File
import scipy.signal as sig # Signal Processing Toolbox 
from scipy.io import wavfile as wav # Read data from and write data to file
from pydub import AudioSegment as AudioSegment # Import and manipulate audio files

def Read_Sound_File(): # Read a Sound File
    
    print("\n")

    # Read Sound File Using Soundfile:
    [data1, Fs] = sn.read('Impact_Moderato.wav')
    data1 = data1[:,1]
    plt.xlabel('Number of Frames (Sampling Frequency * Time Duration)')
    plt.ylabel('Amplitude')
    plt.title('Read Sound File Using Soundfile')
    plt.plot(data1) 
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Read Sound File Using Scipy
    [F1, data2] = sc.read('Impact_Moderato.wav')
    data2 = data2[:,1]
    plt.xlabel('Number of Frames (Sampling Frequency * Time Duration)')
    plt.ylabel('Amplitude')
    plt.title('Read Sound File Using Scipy')
    plt.plot(data2) 
    plt.grid(True) # Configure the grid lines
    plt.show()

def Generate_Sound(): # Generate Sound

    # Give information about available devices and find the default input/output device(s):
    print("\nThe list of audio devices connected to your system is as follows:\n" + str(sd.query_devices()))
    print("\nNote: > indicates the default input device and < indicates the default output device respectively - " + str(sd.default.device))

    # Define the Time Axis:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 16000
    t = int(input("Enter the time duration in seconds: ")) # t = 2
    n = np.arange(0,t,1/Fs)

    # Generate a Sine Wave:
    f1 = 540
    print("\nThe initial frequency has been set to " + str(f1) + " Hertz.")
    x1 = np.sin(2*np.pi*f1*n)
    f2 = 600
    print("\nThe final frequency has been set to " + str(f2) + " Hertz.")
    x2 = np.sin(2*np.pi*f2*n)
    x = x1 + x2;

    # Play back a NumPy array containing audio data:
    sd.play(x,Fs,t)
    
def Simulate_Amplitude_Modulation(): # Simulate Amplitude Modulation

    # Define the Time Axis:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 2000
    t = int(input("Enter the time duration in seconds: ")) # t = 2
    n = np.arange(0,t,1/Fs)

    # Define the Carrier Wave:
    Fc = int(input("\nEnter the carrier signal frequency in Hertz: ")) # Fc = 50
    Ac = 1
    print("The value of amplitude for carrier signal frequency has been set to " + str(Ac) + ".")
    c = Ac*np.cos(2*np.pi*Fc*n) # Carrier Wave

    # Define the Message Signal:
    Fm = int(input("\nEnter the message signal frequency (less than carrier signal frequency) in Hertz: ")) # Fm = 2
    Am = 0.5
    print("\nThe value of amplitude for message signal frequency in under modulation case has been set to " + str(Am) + ".")
    m = Am*np.sin(2*np.pi*Fm*n) # Message Signal
    s = c * (1 + m/Ac) # Amplitude Modulated Signal

    # Plot Under Modulation Case:
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.title('Under Modulation Case')
    plt.plot(n,s)
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Full Modulation Case:
    Am = 1
    print("The value of amplitude for message signal frequency in full modulation case has been set to " + str(Am) + ".")
    m = Am*np.sin(2*np.pi*Fm*n) # Message Signal
    s = c * (1 + m/Ac) # Amplitude modulated signal

    # Plot Full Modulation Case:
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.title('Full Modulation Case')
    plt.plot(n,s,'g')
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Over Modulation Case:
    Am = 1.5
    print("The value of amplitude for message signal frequency in over modulation case has been set to " + str(Am) + ".")
    m = Am*np.sin(2*np.pi*Fm*n) # Message Signal
    s = c * (1 + m/Ac) # Amplitude modulated signal

    # Plot Over Modulation Case:
    plt.xlabel('Time(s)') 
    plt.ylabel('Amplitude')
    plt.title('Over Modulation Case')
    plt.plot(n,s,'r') 
    plt.grid(True) # Configure the grid lines
    plt.show()

def Simulate_Frequency_Modulation(): # Simulate Frequency Modulation

    # Details for Frequency Modulation:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 10000
    Fc = int(input("Enter the carrier signal frequency in Hertz: ")) # Fc = 100
    Fm = int(input("Enter the message signal frequency in Hertz: ")) # Fm = 10
    b = int(input("Enter the modulation index: ")) # b = 5

    # Define the Time Axis:
    n = np.arange(0,0.2,1/Fs)
    m = np.cos(2*np.pi*Fm*n) # Message Signal
    fm = np.cos(2*np.pi*Fc*n + b*np.sin(2*np.pi*Fm*n)) # Frequency Modulated Signal

    # Plot the Signals:
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.plot(n,fm); plt.plot(n,m)
    plt.legend(['Frequency Modulated Signal','Message Signal'])
    plt.grid(True) # Configure the grid lines
    plt.show()

def Simulate_Frequency_Demodulation(): # Simulate Frequency Demodulation

    # Function to plot waveform of an input signal:
    def waveform(path, title):
        
        # Compute waveform of the input signal:
        raw = wave.open(path)
        signal = raw.readframes(-1)
        signal = np.frombuffer(signal, dtype ="int16")
        f_rate = raw.getframerate()
        time = np.linspace(0, len(signal) / f_rate, num = len(signal))
        
        # Plot Input Sound Wave:
        plt.xlabel('Time (in Seconds)') 
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.plot(time, signal)
        plt.grid(True) # Configure the grid lines
        plt.show()
    
    # To print input sound waveform:
    waveform('Symphony_9.wav', 'Input Sound Wave')
    
    def frequency_modulation():
        
        def freq_mod():
            
            input_src = wave.open('Symphony_9.wav', 'r') # Open input audio file 
            fm_carrier = 10000 # Carrier Drequency
            # Sampling Frequency = 48000 Hz
            max_deviation = 1000 # Frequency in Hertz
            
            fm = wave.open('Symphony_9_fm.wav', 'w') # Create file to store resultant wave
            fm.setnchannels(1) # Sets the channel as 1 - mono
            fm.setsampwidth(2) # Sets the sample width to 2 bytes
            fm.setframerate(48000) # Sets the number of samples per second as 48000
            
            phase = 0 # Angle in radians
            
            # Loop runs through all frames of input sound signal:
            for n in range(0, input_src.getnframes()):
                    
                    # Increase or decrease phase according to input signal:
                    inputsgn = struct.unpack('h', input_src.readframes(1))[0] / 32768.0
            
                    # Translate input into a phase change: 
                    phase += inputsgn * math.pi * max_deviation / 48000
                    phase %= 2 * math.pi
            
                    # Calculate quadrature I/Q:
                    i = math.cos(phase)
                    q = math.sin(phase)
            
                    # Calculate the number of frames in the data:
                    carrier = 2 * math.pi * fm_carrier * (n / 48000.0)
                    output = i * math.cos(carrier) - q * math.sin(carrier)            
                    fm.writeframes(struct.pack('h', int(output * 32767)))            
                
        # Driver code to plot the waveform of modulated signal:
        print("\nModulating the input signal...")
        freq_mod() # Function call to freq_mod() to modulate the input signal
        waveform('Symphony_9_fm.wav', 'Modulated Signal') 
         
    def frequency_demodulation(): 
        
        from numpy import fft # FFT computes the one-dimensional DFT
    
        SAMPLE_RATE = 48000 # Frequency in Hertz
        NYQUIST_RATE = SAMPLE_RATE / 2.0 # The minimum rate at which a finite bandwidth signal needs to be sampled to retain all of the information.
        FFT_LENGTH = 512 # The length of the FFT input data frame in samples
    
       # Define lowpass FIR filter to filter out high frequency copies:
        def lowpass_coefs(cutoff):
            
                cutoff /= (NYQUIST_RATE / (FFT_LENGTH / 2.0))
    
                # Create FFT Filter Mask:
                mask = []
                negatives = []
                l = FFT_LENGTH // 2
                for f in range(0, l+1):
                        rampdown = 1.0
                        if f > cutoff:
                                rampdown = 0
                        mask.append(rampdown)
                        if f > 0 and f < l:
                                negatives.append(rampdown)    
                negatives.reverse()
                mask = mask + negatives
    
                # Convert FFT Filter Mask to FIR Coefficients:
                impulse_response = fft.ifft(mask).real.tolist()
    
                # Swap Left and Right Sides:
                left = impulse_response[:FFT_LENGTH // 2]
                right = impulse_response[FFT_LENGTH // 2:]
                impulse_response = right + left    
                b = FFT_LENGTH // 2
                
                # Apply Triangular Window Function:
                for n in range(0, b):
                            impulse_response[n] *= (n + 0.0) / b
                for n in range(b + 1, FFT_LENGTH):
                            impulse_response[n] *= (FFT_LENGTH - n + 0.0) / b
    
                return impulse_response
    
        # Create lowpass filter with cutoff, original as parameters:
        def lowpass(original, cutoff): 
                coefs = lowpass_coefs(cutoff)
                return np.convolve(original, coefs)
            
        def freq_demod():           
            
            input_src = wave.open("Symphony_9_fm.wav", "r") # Open a new wav file
            fm_carrier = 10000.0 # Carrier Frequency
            # Sampling frequency = 48000 Hz
            max_deviation = 1000.0 # Frequency in Hertz
            
            demod = wave.open("Symphony_9_demod.wav", "w") # Create a new wav file
            demod.setnchannels(1) # Sets the channel as 1 - mono
            demod.setsampwidth(2) # Sets the sample width to 2 bytes
            demod.setframerate(48000) # Sets the number of samples per second as 48000
            
            # Generate random floating numbers between 0 and 1:
            initial_carrier_phase = random.random() * 2 * math.pi            
            last_angle = 0.0
            istream = []
            qstream = []
            
            for n in range(0, input_src.getnframes()):
                
                    # Take bytes and convert them to non-byte values:
                    inputsgn = struct.unpack('h', input_src.readframes(1))[0] / 32768.0
            
                    # I/Q demodulation, not unlike QAM:
                    carrier = 2 * math.pi * fm_carrier * (n / 48000.0) + initial_carrier_phase
                    istream.append(inputsgn * math.cos(carrier))
                    qstream.append(inputsgn * -math.sin(carrier))
            
            istream = lowpass(istream, 1500) # Cutoff frequency = 1500 Hertz
            qstream = lowpass(qstream, 1500) # Cutoff frequency = 1500 Hertz            
            last_output = 0
            
            for n in range(0, len(istream)):
                
                    i = istream[n]
                    q = qstream[n]
            
                    # Determine phase (angle) of I/Q pair:
                    angle = math.atan2(q, i)
            
                    # Change of angle = baseband signal:
                    angle_change = last_angle - angle
            
                    # No large phase changes are expected:
                    if angle_change > math.pi:
                            angle_change -= 2 * math.pi
                    elif angle_change < -math.pi:
                            angle_change += 2 * math.pi
                    last_angle = angle
            
                    # Convert angle change to baseband signal strength:
                    output = angle_change / (math.pi * max_deviation / 48000)
                    
                    if abs(output) >= 1:
                        
                            # Some unexpectedly big angle change happened:
                            output = last_output
                            
                    last_output = output
                    
                    # Calculate the number of frames in the data:
                    demod.writeframes(struct.pack('h', int(output * 32767)))
       
        # Driver code to plot the waveform of demodulated signal:
        print("\nDemodulating the modulated signal...")
        freq_demod() # Function call to freq_demod() to demodulate the modulated signal
        waveform('Symphony_9_demod.wav', 'Demodulated Signal')
        
    frequency_modulation()
    frequency_demodulation()

def Record_Sound_Spectral_Analysis(): # Record Sound and Do Spectral Analysis
    
    from scipy.fftpack import fft # Module to calculate discrete fast Fourier transform 

    # Details for Sound Recording:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 16000
    d = int(input("Enter the time duration in seconds: ")) # d = 3

    # Give information about available devices and find the default input/output device(s):
    print("\nThe list of audio devices connected to your system is as follows:\n" + str(sd.query_devices()))
    print("\nNote: > indicates the default input device and < indicates the default output device respectively - " + str(sd.default.device))

    # Record audio data from your sound device into a NumPy array:
    print("\nRecording has started.")
    a = sd.rec(int(d*Fs), Fs, 1, blocking = 'True') 
    #sd.wait()
    a = a.flatten(); # Return a copy of the array (matrix) collapsed into one dimension.
    #t = np.arange(0,d,1/Fs)
    #a = np.sin(2*3.14*2000*t)
    print("Recording has stopped.")

    # Play back a NumPy array containing audio data:
    sd.play(a,Fs)

    # Plot the Recorded Wave:
    plt.xlabel('Number of Frames (Sampling Frequency * Time Duration)')
    plt.ylabel('Amplitude')
    plt.title('Recorded Sound')
    plt.plot(a)
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Fast Fourier Transform Spectrum:
    X_f = fft(a)
    #X_f = fft2(a)

    # Create Frequency Axis:
    n = np.size(a) # Count the number of elements in the array
    fr = (Fs/2)*np.linspace(0,1,round(n/2))
    X_m = (2/n)*abs(X_f[0:np.size(fr)])

    # Plot Magnitude Spectrum:
    plt.xlabel('Frequency (in Hertz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude of Sound Spectrum')
    plt.plot(fr, X_m)
    plt.grid(True) # Configure the grid lines
    plt.show()

def Design_And_Analyze_Filters(): # Design and Analyze Filters
    
    # Define the Time Axis:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 1000
    n = int(input("Enter the filter order for IIR Filter: ")) # n = 5
    fc = np.array([100, 300]) # Cutoff Frequency Array
    print("\n")

    # Design IIR Butterworth Filter:
    w_c = 2*fc/Fs # Normalized Frequency
    [b,a] = sig.butter(n, w_c, btype='bandpass') # Butterworth digital and analog filter design

    # Frequency Response of IIR Filter: 
    [w,h] = sig.freqz(b,a,worN = 2000) # Compute the frequency response of a digital filter
    w = Fs*w/(2*np.pi) # Convert Ω to Hertz
    h_db = 20*np.log10(abs(h)) # Convert magntiude to dB

    # Plot IIR Butterworth Filter:
    plt.xlabel('Frequency (in Hertz)')
    plt.ylabel('Magnitude (in dB)')
    plt.title('IIR Filter Response')
    plt.plot(w, h_db)
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Design FIR Filter:
    N = int(input("Enter the number of coefficients for FIR Filter: ")) # N = 20
    fc = np.array([100, 200]) # Cutoff Frequency Array
    w_c = 2*fc/Fs # Normalized Frequency
    N = int(N | 1) # Set the lowest bit of N to 1 to ensure N is odd.
    t = sig.firwin(N,w_c) # FIR filter design using the window method

    # Frequency Response of FIR Filter: 
    [w,h] = sig.freqz(t, worN = 2000) # Compute the frequency response of a digital filter
    w = Fs*w/(2*np.pi) # Convert Ω to Hertz
    h_db = 20*np.log10(abs(h)) # Convert magntiude to dB

    # Plot FIR Filter:
    plt.xlabel('Frequency (in Hertz)')
    plt.ylabel('Magnitude (in dB)')
    plt.title('FIR Filter Response')
    plt.plot(w,h_db)
    plt.grid(True) # Configure the grid lines
    plt.show()

def Remove_Noise_From_Noisy(): # Remove Noise from a Noisy Signal
    
    import scipy.fftpack as sf # Module to calculate discrete fast Fourier transform

    # Generate a Signal:
    Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 200
    t = int(input("Enter the time duration in seconds: ")) # t = 4
    n = np.arange(0,t,1/Fs) # Return evenly spaced values within a given interval
    f = int(input("Enter the signal frequency in Hertz: ")) # f = 3
    x = np.sin(2*np.pi*f*n)

    print("\n")

    # Generate a Noise: (Draw random samples from a normal distribution)
    y = np.random.normal(0, 0.2, np.size(x)); # Additive Wired Gaussian Noise 
    x = x + y
    plt.title('Histogram Representation - Gaussian Noise')
    plt.hist(y) # Plot a Histogram
    plt.grid(True) # Configure the grid lines
    plt.show()

    # Plot the Noisy Signal:
    plt.subplot(2,1,1)
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.title('Noisy Sinusoidal Wave')
    plt.plot(n,x) 
    plt.tight_layout() # Adjust the padding between and around subplots
    plt.grid(True) # Configure the grid lines

    # Perform Spectral Analysis:
    X_f = abs(sf.fft(x)) # Return the absolute value of the fast Fourier transform
    l = np.size(x)
    fr = (Fs/2)*np.linspace(0,1,int(l/2))
    xl_m = (2/l)*abs(X_f[0:np.size(fr)]); # Compute Magnitude Spectrum

    # Plot Magntiude Spectrum:
    plt.subplot(2,1,2)
    plt.title('Spectrum of Noisy signal')
    plt.xlabel('Frequency (in Hertz)')
    plt.ylabel('Magnitude (in dB)')
    plt.plot(fr,20*np.log10(xl_m))
    plt.tight_layout() # Adjust the padding between and around subplots
    plt.grid(True) # Configure the grid lines

    plt.show() # Display all open figures

    # Create a Band Pass Filter:
    o = 2; # Order of the Filter
    fc = np.array([1,5]) # Define Cutoff Frequency
    wc = 2*fc/Fs; # Normalize Cutoff Frequency to rad/s
    [b,a] = sig.butter(o, wc, btype = 'bandpass') # Design a Butterworth filter design

    # Compute the frequency response of a digital filter:
    [W,h] = sig.freqz(b,a, worN = 1024)
    W = Fs* W/(2*np.pi) # Convert to Hertz

    # Plot Filter Frequency Response:
    plt.subplot(2,1,1)
    plt.xlabel('Frequency (in Hertz)')
    plt.ylabel('Magnitude (in dB)')
    plt.title('Filter Frequency Response')
    plt.plot(W, 20*np.log10(h))
    plt.tight_layout() # Adjust the padding between and around subplots
    plt.grid(True) # Configure the grid lines

    # Filter the Signal:
    x_filt = sig.lfilter(b,a, x) # Filter data along one-dimension with an IIR or FIR filter
    plt.subplot(2,1,2)
    plt.xlabel('Time (in Seconds)') 
    plt.ylabel('Amplitude')
    plt.title('Filtered Signal')
    plt.plot(n,x_filt)
    plt.tight_layout() # Adjust the padding between and around subplots
    plt.grid(True) # Configure the grid lines

    plt.show() # Display all open figures

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

# Main

print("\n\t\t\t\t Amrita Vishwa Vidyapeetham, Coimbatore")
print("\n\t\t\t\t Department of Computer and Communication Engineering (CCE)")
print("\n\t\t\t\t 19CCE281 - Signal Processing Lab")
print("\n\t\t\t\t 2020-24 Batch")
print("\n\t\t\t\t Third Semester")
print("\n\t\t\t\t Term End Project on 'Sound Analysis and Operations'")
print("\n******************************************************************************************************")
print("\n\t\t\t\t Submitted by: Group 3 - Team 4")
print("\n\t\t\t\t Santosh - CB.EN.U4CCE20053")
print("\n\t\t\t\t V Srihari Moorthy - CB.EN.U4CCE20060")
print("\n\t\t\t\t Sudhan Sarvanan - CB.EN.U4CCE20061")
print("\n\t\t\t\t AR. Vishaline - CB.EN.U4CCE20071")
print("\n******************************************************************************************************")

while True:  # This simulates a Do Loop
    choice = int(input(
        "MENU:\n   1. Read a Sound File.\n   2. Generate Sound.\n   3. Simulate Amplitude Modulation\n   4. Simulate Frequency Modulation.\n   5. Simulate Frequency Demodulation.\n   6. Record Sound and Do Spectral Analysis.\n   7. Design and Analyze Filters\n   8. Remove Noise from a Noisy Signal.\n   9. Create Music Using Given Notes.\n   10. Exit\nEnter the number corresponding to the menu to implement the choice: ")) # Menu Based Implementation

    if choice == 1:
        Read_Sound_File() # Read a Sound File
    elif choice == 2:
        Generate_Sound() # Generate Sound
    elif choice == 3:
        Simulate_Amplitude_Modulation() # Simulate Amplitude Modulation
    elif choice == 4:
        Simulate_Frequency_Modulation() # Simulate Frequency Modulation
    elif choice == 5:
        Simulate_Frequency_Demodulation() # Simulate Frequency Demodulation
    elif choice == 6:
        Record_Sound_Spectral_Analysis() # Record Sound and Do Spectral Analysis
    elif choice == 7:
        Design_And_Analyze_Filters() # Design and Analyze Filters
    elif choice == 8:
        Remove_Noise_From_Noisy() # Remove Noise from a Noisy Signal
    elif choice == 9:
        Create_Music_From_Notes() # Create Music Using Given Notes
    elif choice == 10: 
        break  # Exit loop
    else:
        print("Error: Invalid Input! Please try again.")
