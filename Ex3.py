import numpy as np
import math

SIGMA = 3

def singlePdf(t, mean, sd):
    squaredSigma = sd**2
    denom = (2*math.pi*squaredSigma)**.5
    num = math.exp(-((t - mean) ** 2) / (2 * squaredSigma))
    return num / denom


def ker(x):
    pdfValues = np.zeros(50, dtype=float)
    for t in range(50):
        pdfValues[t] = singlePdf(t, x, SIGMA)
    return pdfValues


def predict(X, H):
    signal = np.zeros(50)
    for i in range(len(X)):
        if H[i] < 0:
            return np.full(50, None)
        signal += (H[i] * ker(X[i]))
    return signal


def optimize(Y, k):
    X0 = np.zeros(k)
    H0 = np.zeros(k)

    def RMSE(X, H):
        predicted_y = predict(X, H)
        return np.sqrt(((predicted_y - Y) ** 2).mean())




X = [2, 10, 40]
H = [10, -9, 5]



print(predict(X, H))
