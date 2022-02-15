import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
from scipy.fftpack import fft # Module to calculate discrete fast Fourier transform 
import sounddevice as sd # Play and record NumPy arrays containing audio signals

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
print("Recording is being played...")
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
