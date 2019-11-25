import numpy as np
import math
from scipy import optimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import time
import sys

SIGMA = 3
BASE_PEAK_HEIGHT = 0.1329807601338109
OPTIMIZE1 = 1
OPTIMIZE2 = 2
FAILED_OPTIMIZE1 = 3
SMART_OPTIMIZE1 = 4
SMART_OPTIMIZE2 = 5

PLOT_NOTHING = 0
PLOT_COMPARISON = 1
PLOT_ALL = 2


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


def RMSE(to_optimize, Y):
    half = int(len(to_optimize) / 2)
    X = to_optimize[:half]
    H = to_optimize[half:]

    predicted_y = predict(X, H)
    if predicted_y[0] is None:
        return math.inf
    return np.sqrt(((predicted_y - Y) ** 2).mean())


def failed_smart_optimizer(Y, k):
    local_maxima = argrelextrema(Y, np.greater)  # Search for straight forward peaks
    H0 = np.zeros(k, dtype=int)
    X0 = local_maxima[0][:k]

    for i in range(X0.shape[0]):
        H0[i] = (max(Y[X0[i]] / (BASE_PEAK_HEIGHT * 2), 0))

    if len(H0) < k:
        avrage_h = max(np.mean(Y) / BASE_PEAK_HEIGHT, 0)
        for j in range(len(H0), k):
            H0[j] = avrage_h

    if len(X0) < k:  # If not enough peaks found, generates random locations.
        for i in range(len(X0), k):
            X0 = np.append(X0, np.random.randint(0, 49))

    to_optimize = X0.tolist() + H0.tolist()
    best = optimize.fmin(func=RMSE, x0=to_optimize, args=(Y,), xtol=0.05, full_output=True)
    optimized_params = best[0][:k*2]
    cur_RMSE = best[1]
    return cur_RMSE, optimized_params


def smart_optimize1(Y):
    local_maxima = argrelextrema(Y, np.greater)  # Search for straight forward peaks
    X0 = local_maxima[0]
    max_h = max(Y) / BASE_PEAK_HEIGHT
    H0 = np.random.randint(0, max_h, size=X0.shape[0])

    to_optimize = X0.tolist() + H0.tolist()
    best = optimize.fmin(func=RMSE, x0=to_optimize, args=(Y,), xtol=0.1, full_output=True)
    optimized_params = best[0][:X0.shape[0]*2]
    cur_RMSE = best[1]
    return cur_RMSE, optimized_params


def optimize1(Y, k):
    X0 = np.random.randint(0, 49, size=k)
    max_h = max(Y) / BASE_PEAK_HEIGHT
    H0 = np.random.randint(0, max_h, size=k)

    to_optimize = X0.tolist() + H0.tolist()
    best = optimize.fmin(func=RMSE, x0=to_optimize, args=(Y,), full_output=True)
    optimized_params = best[0][:k*2]
    cur_RMSE = best[1]
    return cur_RMSE, optimized_params


def RMSE2(X, H, Y):
    predicted_y = predict(X, H)
    if predicted_y[0] is None:
        return math.inf
    return np.sqrt(((predicted_y - Y) ** 2).mean())


def optimize2(Y, k):
    prev_RMSE = 0
    cur_RMSE = 1
    x_matrix = np.zeros((k, 50), dtype=float)
    X0 = np.random.randint(0, 49, size=k)

    while abs(cur_RMSE - prev_RMSE) > 0.001:
        prev_RMSE = cur_RMSE
        for i, x in enumerate(X0):
            x_matrix[i] = ker(x)
        H0 = optimize.nnls(x_matrix.T, Y)[0]
        numerically = optimize.fmin(func=RMSE2, x0=X0, args=(H0, Y), full_output=True)
        X0 = numerically[0]
        cur_RMSE = numerically[1]
    return cur_RMSE, X0.tolist() + H0.tolist()


def smart_optimize2(Y):
    local_maxima = argrelextrema(Y, np.greater)  # Search for straight forward peaks
    X0 = local_maxima[0]
    k = X0.shape[0]
    prev_RMSE = 0
    cur_RMSE = 1
    x_matrix = np.zeros((k, 50), dtype=float)

    while abs(cur_RMSE - prev_RMSE) > 0.001:
        prev_RMSE = cur_RMSE
        for i, x in enumerate(X0):
            x_matrix[i] = ker(x)
        H0 = optimize.nnls(x_matrix.T, Y)[0]
        numerically = optimize.fmin(func=RMSE2, x0=X0, args=(H0, Y), full_output=True)
        X0 = numerically[0]
        cur_RMSE = numerically[1]
    return cur_RMSE, X0.tolist() + H0.tolist()


def wraper(genome_pos, Y, k, N, optimization_method, plt_method):
    if optimization_method == OPTIMIZE1:
        results = [optimize1(Y, k) for i in range(N)]
    elif optimization_method == FAILED_OPTIMIZE1:
        results = [failed_smart_optimizer(Y, k) for i in range(N)]
    elif optimization_method == OPTIMIZE2:
        results = [optimize2(Y, k) for i in range(N)]
    elif optimization_method == SMART_OPTIMIZE1:
        results = [smart_optimize1(Y) for i in range(N)]
    else:
        results = [smart_optimize2(Y) for i in range(N)]

    best = min(results, key=lambda item: item[0])

    if plt_method == PLOT_ALL:
        num_of_peaks = int(len(best[1]) / 2)
        predicted_y = predict(best[1][:num_of_peaks], best[1][num_of_peaks:])
        t = range(50)
        fig = plt.figure()
        plt.plot(t, Y, 'r')  # plotting t, a separately
        plt.plot(t, predicted_y, 'b')  # plotting t, b separately
        fig.suptitle('k = {}, genome position = {}'.format(k, genome_pos+1), fontsize=20)
        plt.show()
    return best


def plotResults(RMSE_arr):
    colorsArr = ['b', 'm', 'k', 'c']

    for i in range (len(RMSE_arr)):
        plt.plot(np.arange(1,6),RMSE_arr[i], c=colorsArr[i], label='#{}'.format(i+1), alpha=0.7)
    plt.xlabel("K")
    plt.ylabel("RMSE")
    plt.title("RMSE with Varying K's")
    plt.legend(title="Position")
    # plt.savefig("RMSE_Ks_comp", dpi=200, transparent=True)
    plt.show()


def get_y(file):
    Y = np.zeros((4, 50), dtype=float)
    count = 0
    for row in file:
        Y[0, count] = (float(row.split('\t')[0]))
        Y[1, count] = (float(row.split('\t')[1]))
        Y[2, count] = (float(row.split('\t')[2]))
        Y[3, count] = (float(row.split('\t')[3]))
        count += 1
    return Y


def execute_optimize_with_k(file, optimization_method, plt_method, print_values, timeing):
    Y = get_y(file)
    all_rmse = []
    bests = []
    for i in range(Y.shape[0]):
        if optimization_method < 4 or timeing == 1:
            single_pos_rmse = [(wraper(i, Y[i], k, 10, optimization_method, plt_method)[0]) for k in range(1, 6)]
        else:
            single_pos_rmse = [(wraper(i, Y[i], 0, 10, optimization_method, plt_method)[0])] * 5
        bests.append(min(single_pos_rmse))
        all_rmse.append(single_pos_rmse)
    if print_values == 1:
        print(bests)
    if plt_method > 0:
        plotResults(all_rmse)


"""
Runs Ex3 computational genomics with the next arguments:
argv[1] = genome positions file path, tab format
argv[2] = wanted optimization method:
    1 - optimize1 - random initialization (Q3)
    2 - failed_optimize1 - an "smart" approach of optimize1 initialization. failed.
    3 - optimize2 (Q6)
    4 - BONUS - smart_optimize1
    5 - BONUS - smart_optimize2
argv[3] = plot method:
    0 - Plot nothing
    1 - Plot only one graph that represents all of the performance.
    2 - Plot graph for each Y and predicted_Y for each k.
argv[4] = print values - if equals 1, prints the best RMSE achieved for each position
argv[5] = timeing - if 1, prints the time the program run.
"""
def main():
    genome_positions_path = sys.argv[1]
    optimization_method = int(sys.argv[2])
    plot_method = int(sys.argv[3])
    print_values = int(sys.argv[4])
    timeing = int(sys.argv[5])

    start_time = time.clock()

    f = open(genome_positions_path, "rU")
    execute_optimize_with_k(f, optimization_method, plot_method, print_values, timeing)

    if timeing == 1:
        print(time.clock() - start_time)


if __name__ == "__main__":
    main()
