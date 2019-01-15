# Compute confidence interval for a quantile.
#
# Suppose I'm interested in estimating the 37th percentile.  The
# empirical CDF gives me one estimate for that.  I'd like
# to get a confidence interval: I'm 90% confident that the 37th percentile
# lies between X and Y.
#
# You can compute that with two calls to the following function
# (supposing you're interested in [5%-95%] range) by something like the
# following:
# n = len(sorted_data)
# X_index = CDF_error(n,0.37,0.05)
# Y_index = CDF_error(n,0.37,0.95)
# X=sorted_data[X_index]
# Y=sorted_data[Y_index]
# 90% confidence interval is [X,Y]

# Author: wfbradley
# URL: https://github.com/wfbradley/CDF-confidence/blob/master/CDF_confidence.py
# commit cb0fedcde83f6cede12f5b4ab7e8f05ec258d67a

import numpy as np
from scipy.stats import beta


# The beta distribution is the correct (pointwise) distribution
# across *quantiles* for a given *data point*; if you're not
# sure, this is probably the estimator you want to use. 
def CDF_error_beta(n, target_quantile, quantile_quantile):
	k = target_quantile * n
	return beta.ppf(quantile_quantile, k, n + 1 - k)


# Compute Dvoretzky-Kiefer-Wolfowitz confidence bands.
def CDF_error_DKW_band(n, target_quantile, quantile_quantile):
	# alpha is the total confidence interval size, e.g. 90%.
	alpha = 1.0 - 2.0 * np.abs(0.5 - quantile_quantile)
	epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * float(n)))
	if quantile_quantile < 0.5:
		return max((0, target_quantile - epsilon))
	else:
		return min((1, target_quantile + epsilon))
