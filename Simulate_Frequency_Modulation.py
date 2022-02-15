import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices

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
