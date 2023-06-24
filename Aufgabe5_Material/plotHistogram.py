import cv2
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


img = cv2.imread('schrott.png')
hist = plotHistogram(img, True)
