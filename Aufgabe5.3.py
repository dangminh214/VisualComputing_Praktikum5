import cv2
import numpy as np

# Laden des Bildes
image = cv2.imread('Aufgabe5_Material/kante.png', cv2.IMREAD_GRAYSCALE)

# Definition der Faltungsmatrizen
f1 = np.array([[0, 0, 0],
               [-1, 1, 0],
               [0, 0, 0]])

f2 = np.array([[0, 0, 0],
               [0, -1, 1],
               [0, 0, 0]])

f3 = np.array([[0, 0, 0],
               [1, -2, 1],
               [0, 0, 0]])

f4 = np.array([[0, 0, 0],
               [0.333, 0.333, 0.333],
               [0, 0, 0]])

f5 = np.array([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])

# Definition der Faltungsmatrix F6 als Differenz von F4 und F5
f6 = f4 - f5

# Durchführen der Faltung
result1 = cv2.filter2D(image, -1, f1, delta=128)
result2 = cv2.filter2D(result1, -1, f2, delta=128)
result3 = cv2.filter2D(image.copy(), -1, f3, delta=128)
result4 = cv2.filter2D(image.copy(), -1, f4, delta=128)

# Anzeigen der Ergebnisse
cv2.imshow('Faltung mit F1', result1)
cv2.imshow('Faltung mit F2', result2)
cv2.imshow('Faltung mit F3', result3)
cv2.imshow('Faltung mit F4', result4)

result_f5 = cv2.filter2D(image.copy(), -1, f5, delta=128)

# Durchführen der Faltung mit F4 und Subtraktion von F5
result_f6 = cv2.filter2D(image.copy(), -1, f6, delta=128)

# Anzeigen der Ergebnisse
cv2.imshow('Faltung mit F5', result_f5)
cv2.imshow('Faltung mit F6 (F4 - F5)', result_f6)
cv2.waitKey(0)
cv2.destroyAllWindows()