import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
# imports for sped up hdquantiles_sd
from numpy import float_, int_, ndarray
import numpy.ma as ma
from scipy.stats.distributions import norm, beta, t, binom


def pval_to_asterisk(pval, brackets=True):
    """
    convert p-value scalar or array/dataframe of p-vales to significance strings 'ns', '*', '**', etc.
    :param pval: scalar, array or dataframe of p-values
    :return: scalar or array with the same shape as the input, where each p-value is converted to its significance
    string
    """
    def translate(pval, brackets=True):
        if brackets:
            lb = '('
            rb = ')'
        else:
            lb = ''
            rb = ''
        if pval > 0.05:
            return lb + 'ns' + rb
        elif pval > 0.01:
            return lb + '*' + rb
        elif pval > 0.001:
            return lb + '**' + rb
        elif pval > 0.0001:
            return lb + '***' + rb
        else:
            return lb + '****' + rb
    if np.isscalar(pval):
        return translate(pval, brackets)
    else:
        return np.vectorize(translate)(pval, brackets)

def plot_pvals(pvals, xs, ys, ylim=None, corrected_pvals=None, df_pval_location='above', color=None):
    if ylim is None:
        ylim = plt.gca().get_ylim()
    offset = (np.max(ylim) - np.min(ylim)) / 40  # vertical offset between data point and bottom asterisk
    if corrected_pvals is None:
        corrected_pvals = np.ones(shape=np.array(pvals).shape, dtype=np.float32)
    h = []
    for pval, corrected_pval, x, y in zip(pvals, corrected_pvals, xs, ys):
        str = pval_to_asterisk(pval, brackets=False).replace('*', '∗')
        corrected_str = pval_to_asterisk(corrected_pval, brackets=False).replace('*', '∗')
        if str == 'ns':
            fontsize = 10
        else:
            fontsize = 7
        if corrected_str == 'ns':  # we don't plot 'ns' in corrected p-values, to avoid having asterisks overlapped by 'ns'
            corrected_str = ''
        str = str.replace(corrected_str, '⊛'*len(corrected_str), 1)
        if pval > 0.05:
            rotation = 90
        else:
            rotation = 90
        if df_pval_location == 'above':
            y += offset
            va = 'bottom'
        else:
            y -= 2*offset
            va = 'top'
        h.append(plt.text(x, y + offset, str, ha='center', va=va, color=color, rotation=rotation, fontsize=fontsize))
    return h

def plot_model_coeff(x, df_coeff, df_ci_lo, df_ci_hi, df_pval, ylim=None, df_corrected_pval=None, color=None,
                     df_pval_location='above', label=None):
    if color is None:
        # next colour to be used, according to the colour iterator
        color = next(plt.gca()._get_lines.prop_cycler)['color']
    plt.plot(x, df_coeff, color=color, label=label)
    plt.fill_between(x, df_ci_lo, df_ci_hi, alpha=0.5, color=color)
    h = plot_pvals(df_pval, x, df_ci_hi, corrected_pvals=df_corrected_pval, ylim=ylim, color=color,
                   df_pval_location=df_pval_location)
    return h

def plot_model_coeff_compare2(x, df_coeff_1, df_ci_lo_1, df_ci_hi_1, df_pval_1,
                              df_coeff_2, df_ci_lo_2, df_ci_hi_2, df_pval_2,
                              ylim=None,
                              df_corrected_pval_1=None, df_corrected_pval_2=None,
                              color_1=None, color_2=None,
                              label_1=None, label_2=None):
    if color_1 is None:
        # next colour to be used, according to the colour iterator
        color_1 = next(plt.gca()._get_lines.prop_cycler)['color']
    if color_2 is None:
        color_2 = next(plt.gca()._get_lines.prop_cycler)['color']
    dx = (np.max(x) - np.min(x)) / 60
    plt.plot(x, df_coeff_1, color=color_1, label=label_1)
    plt.plot(x, df_coeff_2, color=color_2, label=label_2)
    plt.fill_between(x, df_ci_lo_1, df_ci_hi_1, alpha=0.5, color=color_1)
    plt.fill_between(x, df_ci_lo_2, df_ci_hi_2, alpha=0.5, color=color_2)
    y = np.maximum(df_ci_hi_1, df_ci_hi_2)
    h1 = plot_pvals(df_pval_1, x - dx, y, corrected_pvals=df_corrected_pval_1, ylim=ylim, color=color_1)
    h2 = plot_pvals(df_pval_2, x + dx, y, corrected_pvals=df_corrected_pval_2, ylim=ylim, color=color_2)

    return h1, h2


def models_coeff_ci_pval(models, extra_hypotheses=None, model_names=None):
    """
    For convenience, extract betas (coefficients), confidence intervals and p-values from a statsmodels model. Each one
    corresponds to one t-test of a hypothesis (where the hypothesis is that the coefficient ~= 0).
    This function also allows to add extra hypotheses (contrasts) to the model. For example, that the sum of two of
    the model's coefficients is ~= 0.

    * Example of a model:

    import statsmodels.api as sm
    model = sm.RLM.from_formula('weight ~ C(sex)', data=df, M=sm.robust.norms.HuberT()).fit()

    * Example of extra hypotheses:

    'Intercept + C(ko_parent)[T.MAT]'

    :param models: List of statsmodels models (see example above).
    :param extra_hypotheses: (def None) String with new hypotheses to t-test in the model (see example above).
    :param model_names: (def None) List of strings with the name of each model. This will become the index in each
    output dataframe.
    :return: df_coeff, df_ci_lo, df_ci_hi, df_pval
    """
    if extra_hypotheses is not None:
        hypotheses_labels = extra_hypotheses.replace(' ', '').split(',')
    df_coeff_tot = pd.DataFrame()
    df_ci_lo_tot = pd.DataFrame()
    df_ci_hi_tot = pd.DataFrame()
    df_pval_tot = pd.DataFrame()
    for model in models:
        # values of coefficients
        df_coeff = pd.DataFrame(data=model.params).transpose()
        # values of coefficient's confidence interval
        df_ci_lo = pd.DataFrame(data=model.conf_int()[0]).transpose()
        df_ci_hi = pd.DataFrame(data=model.conf_int()[1]).transpose().reset_index()
        # p-values
        df_pval = pd.DataFrame(data=model.pvalues).transpose()
        # extra p-values
        if extra_hypotheses is not None:
            extra_tests = model.t_test(extra_hypotheses)

            df = pd.DataFrame(data=[extra_tests.effect], columns=hypotheses_labels)
            df_coeff = pd.concat([df_coeff, df], axis='columns')

            df = pd.DataFrame(data=[extra_tests.conf_int()[:, 0]], columns=hypotheses_labels)
            df_ci_lo = pd.concat([df_ci_lo, df], axis='columns')

            df = pd.DataFrame(data=[extra_tests.conf_int()[:, 1]], columns=hypotheses_labels)
            df_ci_hi = pd.concat([df_ci_hi, df], axis='columns')

            df = pd.DataFrame(data=[extra_tests.pvalue], columns=hypotheses_labels)
            df_pval = pd.concat([df_pval, df], axis='columns')

        df_coeff_tot = pd.concat((df_coeff_tot, df_coeff))
        df_ci_lo_tot = pd.concat((df_ci_lo_tot, df_ci_lo))
        df_ci_hi_tot = pd.concat((df_ci_hi_tot, df_ci_hi))
        df_pval_tot = pd.concat((df_pval_tot, df_pval))

    df_coeff_tot = df_coeff_tot.reset_index()
    df_ci_lo_tot = df_ci_lo_tot.reset_index()
    df_ci_hi_tot = df_ci_hi_tot.reset_index()
    df_pval_tot = df_pval_tot.reset_index()
    df_coeff_tot.drop(labels='index', axis='columns', inplace=True)
    df_ci_lo_tot.drop(labels='index', axis='columns', inplace=True)
    df_ci_hi_tot.drop(labels='index', axis='columns', inplace=True)
    df_pval_tot.drop(labels='index', axis='columns', inplace=True)
    if model_names is not None:
        df_coeff_tot['model'] = model_names
        df_ci_lo_tot['model'] = model_names
        df_ci_hi_tot['model'] = model_names
        df_pval_tot['model'] = model_names
        df_coeff_tot = df_coeff_tot.set_index('model')
        df_ci_lo_tot = df_ci_lo_tot.set_index('model')
        df_ci_hi_tot = df_ci_hi_tot.set_index('model')
        df_pval_tot = df_pval_tot.set_index('model')
    return df_coeff_tot, df_ci_lo_tot, df_ci_hi_tot, df_pval_tot

# likelihood ratio test by Joanna Diong
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
def lrtest(llmin, llmax):
    """
    Likelihood Ratio Test (LRT) by Joanna Diong
    https://scientificallysound.org/2017/08/24/the-likelihood-ratio-test-relevance-and-application/

    Example:

    # import example dataset
    data = sm.datasets.get_rdataset("dietox", "geepack").data

    # fit time only to pig weight
    md = smf.mixedlm("Weight ~ Time", data, groups=data["Pig"])
    mdf = md.fit(reml=False)
    print(mdf.summary())
    llf = mdf.llf

    # fit time and litter to pig weight
    mdlitter = smf.mixedlm("Weight ~ Time + Litter", data, groups=data["Pig"])
    mdflitter = mdlitter.fit(reml=False)
    print(mdflitter.summary())
    llflitter = mdflitter.llf

    lr, p = lrtest(llf, llflitter)
    print('LR test, p value: {:.2f}, {:.4f}'.format(lr, p))

    :param llmin: Log-likelihood of null model (the model without the variable we are considering to add).
    :param llmax: Log-likelihood of the alternative model (the model with the extra variable).
    :return: lr, p
    * lr: likelihood ratio
    * p: p-value to reject the hypothesis that the alternative model fits the data no better than the null model.
    """
    lr = 2 * (llmax - llmin)
    p = stats.chisqprob(lr, 1) # llmax has 1 dof more than llmin
    return lr, p

def plot_linear_regression(model, df, ind_var, other_vars={}, dep_var=None, sx=1.0, tx = 0.0, sy=1.0, ty=0.0,
                           c='C0', marker='x', line_label=''):
    """
    Auxiliary function to make it easier to plot linear regression models. Optionally, also the scatter plot of points
    that the model was computed from.

    We expect a pandas.DataFrame with a column for the independent variable ind_var used to create the model.
    Also, the linear statsmodel model computed from the data. Both ind_var and model will be used to draw a line for the
    linear model.

    In addition, we can select a column in the DataFrame with the dependent variable used to create the model. In that
    case, we also plot the scatter points (ind_var, dep_var) that were used to create the model.

    Finally, for convenience, ind_var and dep_var can be transformed to scale the axes of the plot:
        (ind_var * sx) + tx
        (dep_var * sy) + ty
    This is useful of the variables are standarised, e.g. if
        ind_var = (x - mean(x)) / std(x)

    :param model: statsmodels linear model.
    :param df: pandas.DataFrame that was used to create the model. Note that the x-axis range in the plots is
    (df[ind_var].min(), df[ind_var].max()) * sx + tx.
    :param ind_var: String with the name of the independent variable (x-axis variable).
    :param other_vars: Dictionary with covariates of ind_var in the model, e.g. {'Sex': 'f', 'Genotype': 'WT'}.
    :param dep_var: (def None) String with the name of the dependent variable (y-axis variable) to get a scatter plot of
    the points in df.
    :param sx: (def 1.0) Scaling factor for the independent variable. The value on the x-axis of the plot will be
    (ind_var * sx) + tx.
    :param tx: (def 0.0) Translation factor for the independent variable. The value on the x-axis of the plot will be
    (ind_var * sx) + tx.
    :param sy: (def 1.0) Scaling factor for the dependent variable. The value on the y-axis of the plot will be
    (dep_var * sy) + ty.
    :param ty: (def 0.0) Translation factor for the dependent variable. The value on the y-axis of the plot will be
    (dep_var * sy) + ty.
    :return: None.
    """
    # range for the independent variable
    ind_var_lim = np.array([df[ind_var].min(), df[ind_var].max()])
    vars = {ind_var: ind_var_lim}
    for key in other_vars.keys():
        # duplicate the values provided for the other_vars
        other_vars[key] = [other_vars[key],] * 2
    vars.update(other_vars)
    X = pd.DataFrame(data=vars)
    y_pred = model.predict(X)
    plt.plot(ind_var_lim * sx + tx, y_pred * sy + ty, c, label=line_label)
    if dep_var is not None:
        idx = ~df[ind_var].isna()
        for key, val in other_vars.items():
            idx = idx & (df[key] == val[0])
        plt.scatter(df.loc[idx, ind_var] * sx + tx, df.loc[idx, dep_var] * sy + ty, c=c, marker=marker)
    return None

def inverse_variance_method(x, se):
    """
    Inverse-variance method to combine different estimates of a statistic and their standard errors.

    Let x be a statistic (mean, median, etc) that gets evaluated by different experiments, each with some estimate
    error. If x_i, se_i are the estimate and standard error from the i-th experiment, then the inverse-variance method
    [1, 2] combines the estimates as

    w_i = 1/se_i^2

    x_hat = sum_i (x_i * w_i) / sum_i w_i

    se_hat^2 = 1 / sum_i w_i

    [1] Cochran, W. G. 1937. “Problems Arising in the Analysis of a Series of Similar Experiments.” Supplement to the
    Journal of the Royal Statistical Society 4 (1): 102–18. https://doi.org/10.2307/2984123.

    [2] Cochran, William G. 1954. “The Combination of Estimates from Different Experiments.” Biometrics 10 (1): 101–29.
    https://doi.org/10.2307/3001666.

    :param x: Array of estimates of the statistic. If x is an (N,) array, it gets converted into an (N,1) array. If x is
    a multidimensional array, the operations are conducted over axis=0. For example, if x is a matrix (2D array), each
    column will provide an output value.
    :param se: Array of the corresponding standard errors. Same size as x.
    :return: x_hat, se_hat.
    """

    x = np.array(x)
    se = np.array(se)

    # for code simplicity, if the input is a (N,) array, we turn it into (N,1)
    if x.ndim == 1:
        x = x.reshape((len(x), 1))
    if se.ndim == 1:
        se = se.reshape((len(x), 1))

    # weights: w(i)=1/se(i)^2
    w = 1 / (se ** 2)

    # combined estimate
    sum_w = np.sum(w, axis=0)
    x_hat = np.sum(x * w, axis=0) / sum_w

    # combined standard error
    se_hat = np.sqrt(1 / sum_w)

    return x_hat, se_hat

# originally copied from scipy/stats/mstats_extras.py
# I have edited this to speed it up by a factor of ~537x for a data vector with 60,000 elements
def hdquantiles_sd(data, prob=list([.25,.5,.75]), axis=None):
    """
    The standard error of the Harrell-Davis quantile estimates by jackknife.

    Parameters
    ----------
    data : array_like
        Data array.
    prob : sequence, optional
        Sequence of quantiles to compute.
    axis : int, optional
        Axis along which to compute the quantiles. If None, use a flattened
        array.

    Returns
    -------
    hdquantiles_sd : MaskedArray
        Standard error of the Harrell-Davis quantile estimates.

    See Also
    --------
    hdquantiles

    """
    def _hdsd_1D(data, prob):
        "Computes the std error for 1D arrays."
        xsorted = np.sort(data.compressed())
        n = len(xsorted)

        hdsd = np.empty(len(prob), float_)
        if n < 2:
            hdsd.flat = np.nan

        vv = np.arange(n) / float(n-1)
        betacdf = beta.cdf

        for (i,p) in enumerate(prob):
            _w = betacdf(vv, (n+1)*p, (n+1)*(1-p))
            w = _w[1:] - _w[:-1]
            mx_ = np.fromiter([w[:k] @ xsorted[:k] + w[k:] @ xsorted[k+1:]
                               for k in range(n)], dtype=float_)
            mx_var = np.array(mx_.var(), copy=False, ndmin=1) * n / float(n-1)
            hdsd[i] = float(n-1) * np.sqrt(np.diag(mx_var).diagonal() / float(n))
        return hdsd

    # Initialization & checks
    data = ma.array(data, copy=False, dtype=float_)
    p = np.array(prob, copy=False, ndmin=1)
    # Computes quantiles along axis (or globally)
    if (axis is None):
        result = _hdsd_1D(data, p)
    else:
        if data.ndim > 2:
            raise ValueError("Array 'data' must be at most two dimensional, "
                             "but got data.ndim = %d" % data.ndim)
        result = ma.apply_along_axis(_hdsd_1D, axis, data, p)

    return ma.fix_invalid(result, copy=False).ravel()
