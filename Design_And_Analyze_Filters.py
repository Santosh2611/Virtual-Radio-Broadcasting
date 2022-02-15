import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
import scipy.signal as sig # Signal Processing Toolbox

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
