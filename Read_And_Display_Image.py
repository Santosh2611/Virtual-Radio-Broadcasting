import matplotlib.pyplot as plt # Provides an implicit way of plotting

# Display Image Using imageio
import imageio as img # Provides an easy interface to read and write image data
im_1 = img.imread('Amrita_Vishwa_Vidyapeetham.png') # Loads an image from the specified file
plt.title('Image display using imageio')
plt.imshow(im_1) # Display data as an image, i.e., on a 2D regular raster
plt.axis('off') # Hide the axis (both x-axis & y-axis) in the matplotlib figure
plt.show()

# Display Image Using scikit-image
from skimage import io # Open-source package designed for image preprocessing
im_2 = io.imread('Amrita_Vishwa_Vidyapeetham.png') # Loads an image from the specified file
plt.title('Image display using skit-image')
plt.imshow(im_2) # Display data as an image, i.e., on a 2D regular raster
plt.axis('off') # Hide the axis (both x-axis & y-axis) in the matplotlib figure
plt.show()
