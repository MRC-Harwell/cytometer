
from pathlib import Path
import os
import sys

os.environ['DISPLAY'] = ':1'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import cytometer.stats

# whether to plot and save figures
SAVE_FIGS = False

# script name to identify this experiment
experiment_id = 'arl15del2_exp_0003_phenotyping'

# cross-platform home directory
home = str(Path.home())

sys.path.extend([os.path.join(home, 'Software/cytometer')])

# data paths
root_data_dir = os.path.join(home, 'Data/cytometer_data/arl15del2')
figures_dir = os.path.join(home, 'GoogleDrive/Research/Papers/20211205_Arl15del2_wat_phenotyping/figures')


# load cell data
cell_data_file = os.path.join(root_data_dir, 'Arl15_filtered.csv')
df_all = pd.read_csv(cell_data_file, header=0, sep=',', index_col=False)

# load metainformation
meta_data_file = os.path.join(root_data_dir, 'Arl15-del2 Global KO iWAT and gWAT segmentation analysis.xlsx')
metainfo = pd.read_excel(meta_data_file)

metainfo['Genotype'] = metainfo['Genotype'].astype(
    pd.api.types.CategoricalDtype(categories=['Arl15-Del2:WT', 'Arl15-Del2:Het'], ordered=True))

# rename variables with whitespaces to avoid having to use Q("Age died") syntax everywhere
metainfo = metainfo.rename(columns={'Date of death': 'Date_of_death', 'Date of birth': 'Date_of_birth',
                                    'Age died': 'Age_died', 'Fat mass': 'Fat_mass', 'Lean mass': 'Lean_mass'})

# scale BW to avoid large condition numbers
BW_mean = metainfo['BW'].mean()
metainfo['BW__'] = metainfo['BW'] / BW_mean


## effect of cull age on body weight
########################################################################################################################

bw_model = sm.OLS.from_formula('BW ~ C(Age_died)', data=metainfo).fit()
print(bw_model.summary())
print(bw_model.pvalues)

## effect of genotype on body weight
########################################################################################################################

bw_model = sm.OLS.from_formula('BW ~ C(Genotype)', data=metainfo).fit()
print(bw_model.summary())
print(bw_model.pvalues)

if SAVE_FIGS:

    plt.clf()

    # swarm plot of body weight
    ax = sns.swarmplot(x='Genotype', y='BW', data=metainfo, dodge=True, palette=['C0', 'C1'])
    plt.xlabel('')
    plt.ylabel('Body weight (g)', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.xticks([0, 1], labels=['WT', 'Het'])

    # mean values
    plt.plot([-0.10, 0.10], [bw_model.params['Intercept'], ] * 2, 'k', linewidth=2)
    plt.plot([0.90, 1.10], [bw_model.params['Intercept'] + bw_model.params['C(Genotype)[T.Arl15-Del2:Het]'], ] * 2, 'k',
             linewidth=2)

    # bracket with p-value
    plt.plot([0.0, 0.0, 1.0, 1.0], [60, 62, 62, 60], 'k', lw=1.5)
    pval = bw_model.pvalues['C(Genotype)[T.Arl15-Del2:Het]']
    pval_text = '{0:.3f}'.format(pval) + ' ' + cytometer.stats.pval_to_asterisk(pval)
    plt.text(0.5, 62.5, pval_text, ha='center', va='bottom', fontsize=14)

    plt.ylim(35, 65)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_swarm_bw_genotype.png'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_swarm_bw_genotype.svg'))

## effect of genotype on fat percent and lean mass percent
########################################################################################################################

fatmass_model = sm.OLS.from_formula('Fat_mass ~ BW__ * C(Genotype)', data=metainfo).fit()
print(fatmass_model.summary())

# once we see that the condition number size is due to the scaling of BW, and not collinearity, we recompute the model
fatmass_model = sm.OLS.from_formula('Fat_mass ~ BW * C(Genotype)', data=metainfo).fit()
print(fatmass_model.summary())

leanmass_model = sm.OLS.from_formula('Lean_mass ~ BW__ * C(Genotype)', data=metainfo).fit()
print(leanmass_model.summary())

# once we see that the condition number size is due to the scaling of BW, and not collinearity, we recompute the model
leanmass_model = sm.OLS.from_formula('Lean_mass ~ BW * C(Genotype)', data=metainfo).fit()
print(leanmass_model.summary())

# null models (Genotypes pooled together)
fatmass_model_null = sm.OLS.from_formula('Fat_mass ~ BW', data=metainfo).fit()
leanmass_model_null = sm.OLS.from_formula('Lean_mass ~ BW', data=metainfo).fit()

# compute LRTs and extract p-values and LRs
lrt = pd.DataFrame(columns=['lr', 'pval', 'pval_ast'])

lr, pval = cytometer.stats.lrtest(fatmass_model_null.llf, fatmass_model.llf)
lrt.loc['fatmass_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(leanmass_model_null.llf, leanmass_model.llf)
lrt.loc['leanmass_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# multitest correction using Benjamini-Krieger-Yekutieli
_, lrt['pval_adj'], _, _ = multipletests(lrt['pval'], method='fdr_tsbky', alpha=0.05, returnsorted=False)
lrt['pval_adj_ast'] = cytometer.stats.pval_to_asterisk(lrt['pval_adj'])

# check that just fat mass vs. Genotype doesn't show any effect, so the BW variable is needed
print(sm.OLS.from_formula('Fat_mass ~ Genotype', data=metainfo).fit().summary())

if SAVE_FIGS:
    lrt.to_csv(os.path.join(figures_dir, 'arl15del2_exp_0003_fatmass_leanmass_models_lrt.csv'), na_rep='nan')


if SAVE_FIGS:
    plt.clf()

    plt.subplot(121)
    cytometer.stats.plot_linear_regression(fatmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT'},
                                           dep_var='Fat_mass', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(fatmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het'},
                                           dep_var='Fat_mass', c='C1', marker='o',
                                           line_label='Het')
    plt.xlim(35, 62)
    plt.ylim(17, 37)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Fat mass (g)', fontsize=14)
    plt.legend(loc='upper left')

    plt.subplot(122)
    cytometer.stats.plot_linear_regression(leanmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT'},
                                           dep_var='Lean_mass', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(leanmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het'},
                                           dep_var='Lean_mass', c='C1', marker='o',
                                           line_label='Het')
    plt.xlim(35, 62)
    plt.ylim(12, 25)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Lean mass (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.png'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.jpg'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.svg'))
