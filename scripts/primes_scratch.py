import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from primesieve import *
import pandas as pd
import time

DEBUG = False

# list of precomputed primes
primes_list = np.array(primes(650e3))


def fibonacci(x=0, y=1, n_fibo=10):
    out = [x, y]
    for i in range(n_fibo - 2):
        z = x + y
        x = y
        y = z
        out.append(z)
    return out


def pixel_connectivity(is_prime_square, x_square):
    """
    Count the number of pixels adjacent to each labelled pixel, split into two types of connectivity:
    diagonally and 4-neighbourhood (laterally).
    :param is_prime_square:
    :param x_square:
    :return:
    * pandas.DataFrame
    """

    def is_neigh(is_prime_square, i, j):
        nrow = is_prime_square.shape[0]
        ncol = is_prime_square.shape[1]
        if i < 0 or j < 0 or i >= nrow or j >= ncol:
            return 0
        else:
            return int(is_prime_square[i, j])

    # init outputs
    out_num = []
    out_neigh_d = []
    out_neigh_4 = []

    # loop pixels that correspond to prime numbers
    row, col = np.where(is_prime_square)
    for i, j in zip(row, col):

        # check how many neighbours this pixels is connected to diagonally
        neigh_d = is_neigh(is_prime_square, i - 1, j - 1) \
                  + is_neigh(is_prime_square, i - 1, j + 1) \
                  + is_neigh(is_prime_square, i + 1, j - 1) \
                  + is_neigh(is_prime_square, i + 1, j + 1)

        # check how many neighbours this pixels is connected to laterally
        neigh_4 = is_neigh(is_prime_square, i - 1, j) \
                  + is_neigh(is_prime_square, i + 1, j) \
                  + is_neigh(is_prime_square, i, j - 1) \
                  + is_neigh(is_prime_square, i, j + 1)

        out_num.append(x_square[i, j])
        out_neigh_d.append(neigh_d)
        out_neigh_4.append(neigh_4)

    # put outputs into structured array
    df = pd.DataFrame(data={'num': out_num, 'neigh_d': out_neigh_d, 'neigh_4': out_neigh_4})

    return df



############################################################################################
# example of number square
#
# array([[ 1,  2,  3,  4,  5,  6,  7,  8],
#        [ 9, 10, 11, 12, 13, 14, 15, 16],
#        [17, 18, 19, 20, 21, 22, 23, 24],
#        [25, 26, 27, 28, 29, 30, 31, 32],
#        [33, 34, 35, 36, 37, 38, 39, 40],
#        [41, 42, 43, 44, 45, 46, 47, 48],
#        [49, 50, 51, 52, 53, 54, 55, 56],
#        [57, 58, 59, 60, 61, 62, 63, 64]])
############################################################################################

n = 13
first_number = 105
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# number square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

# compute pixel connectivity
df = pixel_connectivity(is_prime_square, x_square)

############################################################################################
# Loop of Fibonacci number squares
############################################################################################

# number of Fibonacci squares (without counting the initial 0 size)
n_fibo = 16

# side length of each Fibonacci square
fibo_len = fibonacci(0, 1, n_fibo + 1)

# init dataframe with results
df_total = pd.DataFrame([], columns=['i_fibo', 'n', 'is_primes_rank', 'primes_with_neigh_d', 'primes_with_neigh_4',
                                     'time'])

# loop Fibonacci squares
last_number = 1
for i_fibo in range(1, n_fibo + 1):

    # to calculate how long it takes to compute each iteration
    t0 = time.time()

    # i_fibo += 1  ## for manual debugging

    # length of current Fibonacci square
    n = fibo_len[i_fibo]

    # first and last numbers in the square
    first_number = last_number
    last_number = first_number + n**2  # one past the last number, to make use of range() easier

    # list of precomputed primes
    primes_list = np.array(primes(last_number))

    # sequence of numbers contained in the Fibonacci square
    x_list = np.array(range(first_number, last_number))

    # skip the first empty square
    if len(x_list) == 0:
        continue

    # check whether each number is a prime number
    is_prime = np.array([x in primes_list for x in x_list])

    # Fibonacci square filled with the sequence of numbers
    x_square = x_list.reshape((n, n))
    is_prime_square = is_prime.reshape((n, n))

    # plot Fibonacci square
    if DEBUG:
        plt.clf()
        plt.imshow(is_prime_square)
        loc = plticker.MultipleLocator(base=1)
        plt.gca().xaxis.set_major_locator(loc)
        plt.gca().yaxis.set_major_locator(loc)
        plt.grid(which='major', axis='both', linestyle='-')

        for i in range(n):
            for j in range(n):
                plt.text(j, i, '{:d}'.format(x_square[i, j]), color='w', ha='center', va='center', fontsize=int(72/n))

    # compute neighbourhood connectivity
    df = pixel_connectivity(is_prime_square, x_square)

    # skip second square, that has a single non-prime number, so df is empty
    if df.shape[0] == 0:
        continue

    # count how many primes have connectivity 0, 1, 2, 3, 4
    idx, counts = np.unique(df['neigh_d'], return_counts=True)
    counts_d = np.array([0, 0, 0, 0])
    counts_d[idx] = counts
    idx, counts = np.unique(df['neigh_4'], return_counts=True)
    counts_4 = np.array([0, 0, 0, 0])
    counts_4[idx] = counts

    df_total = df_total.append({'i_fibo': i_fibo, 'n': n, 'is_primes_rank': np.linalg.matrix_rank(is_prime_square),
                                'primes_with_neigh_d': counts_d, 'primes_with_neigh_4': counts_4,
                                'time': (time.time() - t0)},
                               ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_total)




# plot column and row patterns
col_pattern = is_prime_square.copy().astype(np.float32)
for j in range(col_pattern.shape[1]):
    col_pattern[:, j] = np.sum(col_pattern[:, j])

row_pattern = is_prime_square.copy().astype(np.float32)
for i in range(row_pattern.shape[0]):
    row_pattern[i, :] = np.sum(row_pattern[i, :])

plt.clf()
plt.subplot(121)
plt.imshow(row_pattern)
plt.subplot(122)
plt.imshow(col_pattern)

plt.clf()
plt.subplot(121)
plt.imshow(row_pattern > 0)
plt.subplot(122)
plt.imshow(col_pattern > 0)
