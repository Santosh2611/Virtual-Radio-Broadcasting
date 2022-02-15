import numpy as np # Support for large, multi-dimensional arrays and matrices
import sounddevice as sd # Play and record NumPy arrays containing audio signals

# Give information about available devices and find the default input/output device(s):
print("\nThe list of audio devices connected to your system is as follows:\n" + str(sd.query_devices()))
print("\nNote: > indicates the default input device and < indicates the default output device respectively - " + str(sd.default.device))

# Define the Time Axis:
Fs = int(input("\nEnter the sampling frequency in Hertz: ")) # Fs = 16000
t = int(input("Enter the time duration in seconds: ")) # t = 1
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
