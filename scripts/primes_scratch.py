"""
This file is part of Cytometer
Copyright 2021 Medical Research Council
SPDX-License-Identifier: Apache-2.0
Author: Ramon Casero <rcasero@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from primesieve import *
import pandas as pd
import time

DEBUG = False

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


def diagonal_zigzag_square(x):
    """
    Fill square matrix in diagonal order.

    x = [0, ..., 8]

    y = diagonal_zigzag_square(x)

    y = [[0 2 5]
         [1 4 7]
         [3 6 8]]
    :param x: vector of length n**2.
    :return: y
    """
    x = np.array(x)
    # length of the square side
    n = np.sqrt(len(x))
    if n != int(n):
        raise ValueError('Input vector needs to have n**2 elements')
    n = int(n)

    # init output matrix
    y = np.zeros((n, n), dtype=x.dtype)

    # number of elements in each diagonal
    tot_diag = list(range(1, n)) + list(range(n, 0, -1))

    # indices in x of the different segments that correspond to each diagonal
    idx = np.cumsum([0] + tot_diag)

    # loop diagonals
    for k in range(2 * n - 1):
        first_i = np.min((k, n-1))
        i = np.array(range(first_i, first_i - tot_diag[k], -1))
        j = k - i
        # print(str(i_all) + ' -> ' + str(j_all))
        y[i, j] = x[idx[k]:idx[k+1]]

    return y


def prop_primes(x):
    """
    Compute proportion of prime numbers in each row/column (even length squares) or diagonal/antidiagonal (odd length
    squares).

    :param x:
    :return:
    """

    # length of square length
    n = x.shape[0]

    if n % 2 == 1:  # odd length square

        prop_fw = []
        for k in range(n - 1, -n, -1):
            diag = np.diagonal(x, k)
            prop_fw.append(np.count_nonzero(diag) / len(diag))

        x = np.fliplr(x)
        prop_bk = []
        for k in range(n - 1, -n, -1):
            diag = np.diagonal(x, k)
            prop_bk.append(np.count_nonzero(diag) / len(diag))

    else:  # even length square

        prop_fw = np.sum(x, axis=1) / n  # rows
        prop_bk = np.sum(x, axis=0) / n  # columns

    if DEBUG:
        if n % 2 == 1:  # odd length square

            plt.clf()
            (markers, stemlines, baseline) = plt.stem(range(n - 1, -n, -1), prop_fw, label='Diagonals')
            plt.setp(markers, marker='D', markersize=6, markeredgecolor="C0", markeredgewidth=2)
            plt.stem(range(n - 1, -n, -1), prop_bk, linefmt='C1', markerfmt='C1o', label='Antidiagonals')
            plt.xlabel('Diagonal index')
            plt.ylabel('Primes proportion')
            plt.legend()

        else:

            plt.clf()
            (markers, stemlines, baseline) = plt.stem(prop_fw, label='Rows')
            plt.setp(markers, marker='D', markersize=6, markeredgecolor="C0", markeredgewidth=2)
            plt.stem(prop_bk, linefmt='C1', markerfmt='C1o', label='Columns')
            plt.xlabel('Row/column index')
            plt.ylabel('Primes proportion')
            plt.legend()

    return np.array(prop_fw), np.array(prop_bk)

# list of precomputed primes
primes_list = np.array(primes(650e3))


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
# Loop of Fibonacci number squares. Fill all the squares row by row
############################################################################################

# number of Fibonacci squares (without counting the initial 0 size)
n_fibo = 16

# side length of each Fibonacci square
fibo_len = fibonacci(0, 1, n_fibo + 1)

# init dataframe with results
df_total = pd.DataFrame([], columns=['i_fibo', 'n', 'rank', 'primes_neigh_d', 'primes_neigh_4',
                                     'prop_fw', 'prop_bk', 'time'])

# for debugging purposes, this can be used to directly get the initial parameters for any matrix, without having to
# check for primes, etc.
if DEBUG:
    n_fibo_to_stop = 15
    last_number = 1
    for i_fibo in range(1, n_fibo_to_stop + 1):
        # length of current Fibonacci square
        n = fibo_len[i_fibo]

        # first and last numbers in the square
        first_number = last_number
        last_number = first_number + n ** 2  # one past the last number, to make use of range() easier

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

    # compute proportion of primes in each diagonal or lateral line
    prop_fw, prop_bk = prop_primes(is_prime_square)

    if DEBUG:
        if n % 2 == 1:  # odd length square

            plt.clf()
            (markers, stemlines, baseline) = plt.stem(range(n - 1, -n, -1), prop_fw, label='Diagonals')
            plt.setp(markers, marker='D', markersize=6, markeredgecolor="C0", markeredgewidth=2)
            plt.stem(range(n - 1, -n, -1), prop_bk, linefmt='C1', markerfmt='C1o', label='Antidiagonals')
            plt.xlabel('Diagonal index')
            plt.ylabel('Primes proportion')
            plt.legend()

        else:

            plt.clf()
            (markers, stemlines, baseline) = plt.stem(prop_fw, label='Rows')
            plt.setp(markers, marker='D', markersize=6, markeredgecolor="C0", markeredgewidth=2)
            plt.stem(prop_bk, linefmt='C1', markerfmt='C1o', label='Columns')
            plt.xlabel('Row/column index')
            plt.ylabel('Primes proportion')
            plt.legend()

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

    df_total = df_total.append({'i_fibo': i_fibo, 'n': n, 'rank': np.linalg.matrix_rank(is_prime_square),
                                'primes_neigh_d': counts_d, 'primes_neigh_4': counts_4,
                                'prop_fw': prop_fw, 'prop_bk': prop_bk,
                                'time': (time.time() - t0)},
                               ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_total)

############################################################################################
# Loop of Fibonacci number squares. Fill the squares with odd length in diagonal order.
#
# This experiment fails. Odd-Fibonacci squares now have primes touching both laterally and
# diagonally.
############################################################################################

# number of Fibonacci squares (without counting the initial 0 size)
n_fibo = 16

# side length of each Fibonacci square
fibo_len = fibonacci(0, 1, n_fibo + 1)

# init dataframe with results
df_total = pd.DataFrame([], columns=['i_fibo', 'n', 'rank', 'primes_neigh_d', 'primes_neigh_4',
                                     'time'])

# for debugging purposes, this can be used to directly get the initial parameters for any matrix, without having to
# check for primes, etc.
if DEBUG:
    n_fibo_to_stop = 9
    last_number = 1
    for i_fibo in range(1, n_fibo_to_stop + 1):
        # length of current Fibonacci square
        n = fibo_len[i_fibo]

        # first and last numbers in the square
        first_number = last_number
        last_number = first_number + n ** 2  # one past the last number, to make use of range() easier

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
    if n % 2 == 1:  # Fibonacci square's length is odd
        x_square = diagonal_zigzag_square(x_list)
        is_prime_square = diagonal_zigzag_square(is_prime)
    else:  # Fibonacci square's length is even
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

    df_total = df_total.append({'i_fibo': i_fibo, 'n': n, 'rank': np.linalg.matrix_rank(is_prime_square),
                                'primes_neigh_d': counts_d, 'primes_neigh_4': counts_4,
                                'time': (time.time() - t0)},
                               ignore_index=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_total)

############################################################################################
# Position of prime numbers within fibonacci intervals
############################################################################################


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


n_fibo = 30
fibo = fibonacci(0, 1, n_fibo + 1)
primes_list = np.array(primes(fibo[-1]))
primes_lot = np.zeros(shape=(fibo[-1],), dtype=np.bool)
primes_lot[primes_list] = True

p_out = []
i_out = []
for i in range(len(fibo) - 1):
    fibo_from = fibo[i]
    fibo_to = fibo[i+1]

    # normalize primes to the interval
    p = np.where(primes_lot[fibo_from:fibo_to])[0] / (fibo_to - fibo_from)

    # append to output vector
    i_out = i_out + [i, ] * len(p)
    p_out = np.concatenate((p_out, p))

plt.clf()
plt.hist(p_out, bins=21)

# interpret as polar numbers
x, y = pol2cart(i_out, p_out * 2 * np.pi)
xmin = np.min(x)
xmax = np.max(x)
ymin = np.min(y)
ymax = np.max(y)

plt.clf()
plt.plot([0, 0], [ymin, ymax], 'k')
plt.plot([xmin, xmax], [0, 0], 'k')
plt.scatter(x, y, s=20, color='C1')
