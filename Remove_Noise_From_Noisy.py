import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
import scipy.fftpack as sf # Module to calculate discrete fast Fourier transform 
import scipy.signal as sig # Signal Processing Toolbox

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
