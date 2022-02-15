import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
import wave as wave # Provides a convenient interface to the WAV sound format
import math as math # Provides access to the mathematical functions
import struct as struct # Used in handling binary data stored in files
import random as random # Implements pseudo-random number generators

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

Simulate_Frequency_Demodulation() # Simulate Frequency Demodulation
