import matplotlib.pyplot as plt # Provides an implicit way of plotting
import soundfile as sn # Read and write sound files
import scipy.io.wavfile as sc # Open a Waveform Audio File (WAV) Format File
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
