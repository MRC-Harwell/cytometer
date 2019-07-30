import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from primesieve import *


# list of precomputed primes
primes_list = np.array(primes(650e3))


############################################################################################
# 8x8 Fibonacci with 41 - 104 integers
############################################################################################

n = 8
first_number = 41
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

############################################################################################
# 13x13 Fibonacci with 105 - 274 integers
############################################################################################

n = 13
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

############################################################################################
# 21x21 Fibonacci with 275 - 714 integers
############################################################################################

n = 21
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

############################################################################################
# 34x34 Fibonacci with 715 - 1871 integers
############################################################################################

n = 34
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

############################################################################################
# 55x55 Fibonacci with 1871 - 4895 integers
############################################################################################

n = 55
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
loc = plticker.MultipleLocator(base=1)
plt.gca().xaxis.set_major_locator(loc)
plt.gca().yaxis.set_major_locator(loc)
plt.grid(True, which='both')

############################################################################################
# 89x89 Fibonacci with 4896 - 12816 integers
############################################################################################

n = 89
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)
# loc = plticker.MultipleLocator(base=1)
# plt.gca().xaxis.set_major_locator(loc)
# plt.gca().yaxis.set_major_locator(loc)
# plt.grid(True, which='both')

############################################################################################
# 144x144 Fibonacci with 12817 - 33552 integers
############################################################################################

n = 144
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)

############################################################################################
# 233x233 Fibonacci with 33553 - 87841 integers
############################################################################################

n = 233
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)

############################################################################################
# 377x377 Fibonacci with 87842 - 229970 integers
############################################################################################

n = 377
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)

############################################################################################
# 610x610 Fibonacci with 229971 - 602070 integers
############################################################################################

n = 610
first_number = last_number
last_number = first_number + n**2  # one past the last number, to make use of range() easier

# sequence of numbers contained in the Fibonacci square
x_list = np.array(range(first_number, last_number))

# check whether each number is a prime number
is_prime = np.array([x in primes_list for x in x_list])

# Fibonacci square with sequence of numbers
x_square = x_list.reshape((n, n))
is_prime_square = is_prime.reshape((n, n))

plt.clf()
plt.imshow(is_prime_square)

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
