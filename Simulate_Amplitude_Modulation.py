import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices

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
s = c * (1 + m/Ac) # Amplitude Modulated signal

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
s = c * (1 + m/Ac) # Amplitude Modulated signal

# Plot Over Modulation Case:
plt.xlabel('Time(s)') 
plt.ylabel('Amplitude')
plt.title('Over Modulation Case')
plt.plot(n,s,'r') 
plt.grid(True) # Configure the grid lines
plt.show()
