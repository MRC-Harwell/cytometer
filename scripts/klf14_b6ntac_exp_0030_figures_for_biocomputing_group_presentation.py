import numpy as np
import matplotlib.pyplot as plt

area = np.linspace(0, 20000, 1000)

# plot synthetic examples of ECDFs
plt.clf()
plt.plot(area, -100 + 200 / (1 + np.exp(- area / 2000)))
plt.plot(area, -100 + 200 / (1 + np.exp(- area / 5000)))
plt.legend(['PAT', 'MAT'])
plt.xlabel(r'Cell area ($\mu m^2$)', fontsize=18)
plt.ylabel('ECDF (%)', fontsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

# areas for the 50% percentile
perc = 50
a_pat = -2000 * np.log(200 / (100 + perc) - 1)
a_mat = -5000 * np.log(200 / (100 + perc) - 1)

plt.plot([0, 20000], [perc, perc], 'k')
plt.plot([0, 20000], [perc, perc], 'k')
plt.plot([a_pat, a_pat], [0, perc], 'k')
plt.plot([a_mat, a_mat], [0, perc], 'k')

# double arrow between area values
plt.annotate(s='', xy=(a_mat, 20), xytext=(a_pat, 20), arrowprops=dict(arrowstyle='<-'))
plt.text(2300, 10, r'$\Delta$area=', fontsize=14)
plt.text(2300, 5, str((a_pat - a_mat) / a_mat * 100) + '%', fontsize=14)

