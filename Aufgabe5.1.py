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

img = cv2.imread('Aufgabe5_Material/schrott.png')
scaled_img = np.clip(img, 100, 150)

# Ermitteln Sie den minimalen und maximalen Grauwert des skalierten Bildes
min_value = np.min(scaled_img)
max_value = np.max(scaled_img)

# Skalieren Sie das Bild auf den vollen Kontrastumfang [0...255]
max_contrast_img = ((scaled_img - min_value) / (max_value - min_value)) * 255

hist = plotHistogram(img, False)
plotHistogram(scaled_img, False)

cv2.imshow("original image", img)
cv2.imshow("max contrast image", max_contrast_img)

cv2.waitKey(0)
cv2.destroyAllWindows()




