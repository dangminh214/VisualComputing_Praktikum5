import cv2
import numpy as np
from matplotlib import pyplot as plt
def plotHistogram(image, cumulative=False):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    if cumulative:
        hist = hist.cumsum()
    hist /= hist.sum()

    plt.figure()
    plt.plot(hist)
    plt.show()
    return hist

def histogram_equalization(image):
    # Step 1: Compute histogram
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Step 2: Compute cumulative histogram
    cumulative_hist = hist.cumsum()

    # Step 3: Compute cumulative distribution function (CDF)
    cdf = cumulative_hist / cumulative_hist[-1]

    # Step 4: Scale CDF to the highest possible pixel intensity (usually 255)
    cdf_scaled = (cdf * 255).astype(np.uint8)

    # Step 5: Apply the resulting mapping to each pixel of the input image
    equalized_image = cdf_scaled[image]

    return equalized_image


# Load the input image
input_image = cv2.imread('Aufgabe5_Material/schrott.png', 0)  # Load as grayscale

# Apply histogram equalization
equalized_image = histogram_equalization(input_image)

# Display the original and equalized images
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')

plotHistogram(input_image, True);
plotHistogram(equalized_image, True);

plt.show()
