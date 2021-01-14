import numpy as np
import matplotlib.pyplot as plt


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
