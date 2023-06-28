import cv2
from matplotlib import pyplot as plt
import numpy as np

def plotHistogram(image, cumulative=False):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    if cumulative:
        hist = hist.cumsum()
    hist /= hist.sum()

    plt.figure()
    plt.plot(hist)
    plt.show()
    return hist

img = cv2.imread('Aufgabe5_Material/schrott.png', cv2.IMREAD_GRAYSCALE)


# min and max contrast value of original image
min_value = np.min(img)
max_value = np.max(img)

# scale to range 100 - 150
scaled_img = ((img - min_value) / (max_value - min_value)) * 50 + 100

# update min max values to scaled image instead of original image
min_value = np.min(scaled_img)
max_value = np.max(scaled_img)

# scale all of them up to 255
max_contrast_img = ((scaled_img - min_value) / (max_value - min_value)) * 255

plotHistogram(img, False)
plotHistogram(scaled_img.astype(np.uint8), False)
plotHistogram(max_contrast_img.astype(np.uint8), False)

cv2.imshow("original image", img)
cv2.imshow("scaled image", scaled_img.astype(np.uint8))
cv2.imshow("max contrast image", max_contrast_img.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()