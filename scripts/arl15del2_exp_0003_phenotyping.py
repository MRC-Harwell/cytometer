
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
import scipy.stats as stats
import seaborn as sns
import cytometer.stats
import cytometer.data

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
    ax = sns.swarmplot(x='Genotype', y='BW', data=metainfo, dodge=True, palette=['C0', 'C1'], s=10)
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

print(fatmass_model_null.summary())
print(leanmass_model_null.summary())

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

    plt.gcf().set_size_inches([6.4, 2.4])

    plt.subplot(121)
    cytometer.stats.plot_linear_regression(fatmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT'},
                                           dep_var='Fat_mass', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(fatmass_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het'},
                                           dep_var='Fat_mass', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(fatmass_model_null, metainfo, 'BW',
                                           c='k--', line_label='All')
    plt.xlim(35, 62)
    plt.ylim(17, 37)
    plt.tick_params(labelsize=14)
    plt.title('Fat mass', fontsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Weight (g)', fontsize=14)
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
    cytometer.stats.plot_linear_regression(leanmass_model_null, metainfo, 'BW',
                                           c='k--', line_label='All')
    plt.xlim(35, 62)
    plt.ylim(12, 25)
    plt.tick_params(labelsize=14)
    plt.title('Lean mass', fontsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.png'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.jpg'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_fatmass_leanmass_models.svg'))

## effect of genotype and BW on depot weight
########################################################################################################################

gwat_model = sm.OLS.from_formula('gWAT ~ BW__ * C(Genotype)', data=metainfo).fit()
print(gwat_model.summary())

gwat_model = sm.OLS.from_formula('gWAT ~ BW * C(Genotype)', data=metainfo).fit()
print(gwat_model.summary())

iwat_model = sm.OLS.from_formula('iWAT ~ BW__ * C(Genotype)', data=metainfo).fit()
print(iwat_model.summary())

iwat_model = sm.OLS.from_formula('iWAT ~ BW * C(Genotype)', data=metainfo).fit()
print(iwat_model.summary())

# null models (Genotypes pooled together)
gwat_model_null = sm.OLS.from_formula('gWAT ~ BW', data=metainfo).fit()
iwat_model_null = sm.OLS.from_formula('iWAT ~ BW', data=metainfo).fit()

print(gwat_model_null.summary())
print(iwat_model_null.summary())

# compute LRTs and extract p-values and LRs
lrt = pd.DataFrame(columns=['lr', 'pval', 'pval_ast'])

lr, pval = cytometer.stats.lrtest(gwat_model_null.llf, gwat_model.llf)
lrt.loc['gwat_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(iwat_model_null.llf, iwat_model.llf)
lrt.loc['iwat_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# multitest correction using Benjamini-Krieger-Yekutieli
_, lrt['pval_adj'], _, _ = multipletests(lrt['pval'], method='fdr_tsbky', alpha=0.05, returnsorted=False)
lrt['pval_adj_ast'] = cytometer.stats.pval_to_asterisk(lrt['pval_adj'])

# check that just fat mass vs. Genotype doesn't show any effect, so the BW variable is needed
print(sm.OLS.from_formula('gWAT ~ Genotype', data=metainfo).fit().summary())
print(sm.OLS.from_formula('iWAT ~ Genotype', data=metainfo).fit().summary())

# get a p-value for the slope of the inguinal DW model for Hets
model_names = ['iwat_model']
extra_hypotheses = 'BW+BW:C(Genotype)[T.Arl15-Del2:Het]'

df_coeff, df_ci_lo, df_ci_hi, df_pval = \
    cytometer.stats.models_coeff_ci_pval(
        [iwat_model],
        extra_hypotheses=extra_hypotheses,
        model_names=model_names)

if SAVE_FIGS:
    plt.clf()

    plt.gcf().set_size_inches([6.4, 2.4])

    plt.subplot(121)
    cytometer.stats.plot_linear_regression(gwat_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT'},
                                           dep_var='gWAT', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het'},
                                           dep_var='gWAT', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(gwat_model_null, metainfo, 'BW',
                                           c='k--', line_label='All')
    plt.xlim(35, 62)
    plt.ylim(1.5, 3.0)
    plt.tick_params(labelsize=14)
    plt.title('Gonadal', fontsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)
    plt.ylabel('Depot weight (g)', fontsize=14)
    plt.legend(loc='upper left')

    plt.subplot(122)
    cytometer.stats.plot_linear_regression(iwat_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT'},
                                           dep_var='iWAT', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(iwat_model, metainfo, 'BW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het'},
                                           dep_var='iWAT', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(iwat_model_null, metainfo, 'BW',
                                           c='k--', line_label='All')
    plt.title('Inguinal', fontsize=14)
    plt.xlim(35, 62)
    plt.ylim(1.1, 2.2)
    plt.tick_params(labelsize=14)
    plt.xlabel('Body weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_dw_models.png'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_dw_models.jpg'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_dw_models.svg'))

## effect of genotype and DW on cell area quartiles
########################################################################################################################

# compute cell quartiles for each mouse

# (only mode, 25%-, 50%- and 75%-quantiles for illustration purposes and debugging)
# 0.05, 0.1 , 0.15, 0.2, ..., 0.9 , 0.95
quantiles = np.linspace(0, 1, 21)  #

# indices of the quantiles we are going to model
i_q1, i_q2, i_q3 = [5, 10, 15]  # Q1, Q2, Q3

# extract ID (38.1e) from Animal (ARL15-DEL2-EM1-B6N/38.1e), so that we can search for the ID in the histology file name
metainfo['id'] = [x.split('/')[-1] for x in metainfo['Animal']]

# create dataframe with one row per mouse/depot, and the area quantiles
df_slides = pd.DataFrame()
slide_names = [x.lower() for x in df_all.columns]
for i in range(metainfo.shape[0]):

    print('Mouse: ' + metainfo.loc[i, 'Animal'])

    for depot in ['gwat', 'iwat']:

        print('\tDepot: ' + depot)

        # get list of all the columns of cell areas that correspond to this mouse/depot
        i_histo = [(metainfo.loc[i, 'id'] in x) and (depot in x) for x in slide_names]

        [print('\t\tslide: ' + x) for x in df_all.columns[i_histo]]

        # concatenate all the cells for this animal
        areas_all = df_all[df_all.columns[i_histo]].to_numpy().flatten()
        areas_all = areas_all[~np.isnan(areas_all)]

        # compute quantiles of the pooled cell population
        areas_at_quantiles = stats.mstats.hdquantiles(areas_all, prob=quantiles, axis=0)

        # name of the histology file, converted to lowercase so that e.g. 38.1E is identified as mouse 38.1e
        # we can use the same slide name for all slides, because they all have the same mouse ID and depot tag
        histo_string = df_all.columns[i_histo][0].lower()

        df_row = cytometer.data.tag_values_with_mouse_info(metainfo=metainfo, s=histo_string,
                                                           values=[areas_at_quantiles[i_q1]], values_tag='area_Q1',
                                                           tags_to_keep=['id', 'Genotype', 'gWAT', 'iWAT'])
        df_row['area_Q2'] = areas_at_quantiles[i_q2]
        df_row['area_Q3'] = areas_at_quantiles[i_q3]

        # check whether the slide is gWAT or iWAT
        if 'gwat' in histo_string:
            df_row['depot'] = 'gWAT'
            df_row['DW'] = df_row['gWAT']
        elif 'iwat' in histo_string:
            df_row['depot'] = 'iWAT'
            df_row['DW'] = df_row['iWAT']
        else:
            raise ValueError('Histology slide cannot be identified as either gWAT or iWAT')

        df_slides = df_slides.append(df_row, ignore_index=True)

# pearson correlation coefficients
rho_df = pd.DataFrame()
rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q1', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q1_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q1', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q1_het', 'rho': rho}, ignore_index=True)

rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q2', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q2_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q2', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q2_het', 'rho': rho}, ignore_index=True)

rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q3', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q3_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'gWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q3', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'gwat_q3_het', 'rho': rho}, ignore_index=True)

rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q1', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q1_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q1', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q1_het', 'rho': rho}, ignore_index=True)

rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q2', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q2_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q2', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q2_het', 'rho': rho}, ignore_index=True)

rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:WT')][['area_Q3', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q3_wt', 'rho': rho}, ignore_index=True)
rho = df_slides[(df_slides['depot'] == 'iWAT') & (df_slides['Genotype'] == 'Arl15-Del2:Het')][['area_Q3', 'DW']].corr().iloc[0, 1]
rho_df = rho_df.append({'model': 'iwat_q3_het', 'rho': rho}, ignore_index=True)

# fit models of area quartiles vs. depot weight * genotype
gwat_q1_model = sm.OLS.from_formula('area_Q1 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()
gwat_q2_model = sm.OLS.from_formula('area_Q2 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()
gwat_q3_model = sm.OLS.from_formula('area_Q3 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()

iwat_q1_model = sm.OLS.from_formula('area_Q1 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()
iwat_q2_model = sm.OLS.from_formula('area_Q2 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()
iwat_q3_model = sm.OLS.from_formula('area_Q3 ~ DW * C(Genotype)', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()

# null models
gwat_q1_model_null = sm.OLS.from_formula('area_Q1 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()
gwat_q2_model_null = sm.OLS.from_formula('area_Q2 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()
gwat_q3_model_null = sm.OLS.from_formula('area_Q3 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'gWAT').fit()

iwat_q1_model_null = sm.OLS.from_formula('area_Q1 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()
iwat_q2_model_null = sm.OLS.from_formula('area_Q2 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()
iwat_q3_model_null = sm.OLS.from_formula('area_Q3 ~ DW', data=df_slides,
                                    subset=df_slides['depot'] == 'iWAT').fit()

print(gwat_q1_model.summary())
print(gwat_q2_model.summary())
print(gwat_q3_model.summary())

print(iwat_q1_model.summary())
print(iwat_q2_model.summary())
print(iwat_q3_model.summary())

print(gwat_q1_model_null.summary())
print(gwat_q2_model_null.summary())
print(gwat_q3_model_null.summary())

print(iwat_q1_model_null.summary())
print(iwat_q2_model_null.summary())
print(iwat_q3_model_null.summary())

# compute LRTs and extract p-values and LRs
lrt = pd.DataFrame(columns=['lr', 'pval', 'pval_ast'])

lr, pval = cytometer.stats.lrtest(gwat_q1_model_null.llf, gwat_q1_model.llf)
lrt.loc['gwat_q1_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(gwat_q2_model_null.llf, gwat_q2_model.llf)
lrt.loc['gwat_q2_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(gwat_q3_model_null.llf, gwat_q3_model.llf)
lrt.loc['gwat_q3_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(iwat_q1_model_null.llf, iwat_q1_model.llf)
lrt.loc['iwat_q1_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(iwat_q2_model_null.llf, iwat_q2_model.llf)
lrt.loc['iwat_q2_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

lr, pval = cytometer.stats.lrtest(iwat_q3_model_null.llf, iwat_q3_model.llf)
lrt.loc['iwat_q3_model', :] = (lr, pval, cytometer.stats.pval_to_asterisk(pval))

# multitest correction using Benjamini-Krieger-Yekutieli
_, lrt['pval_adj'], _, _ = multipletests(lrt['pval'], method='fdr_tsbky', alpha=0.05, returnsorted=False)
lrt['pval_adj_ast'] = cytometer.stats.pval_to_asterisk(lrt['pval_adj'])

if SAVE_FIGS:
    plt.clf()

    plt.gcf().set_size_inches([6.4, 7.2])

    plt.subplot(321)
    cytometer.stats.plot_linear_regression(gwat_q1_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q1', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_q1_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q1', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(gwat_q1_model_null, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.5, 3.0)
    plt.ylim(6.0, 10.5)
    plt.title('Gonadal', fontsize=14)
    plt.ylabel('Area$_{Q1}$ ($\cdot 10^3 \mu m^2$)', fontsize=14)

    plt.subplot(322)
    cytometer.stats.plot_linear_regression(iwat_q1_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q1', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(iwat_q1_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q1', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(iwat_q1_model_null, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.2, 2.1)
    plt.ylim(6.0, 10.5)
    plt.title('Inguinal', fontsize=14)
    plt.legend(loc='lower right')

    plt.subplot(323)
    cytometer.stats.plot_linear_regression(gwat_q2_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q2', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_q2_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q2', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(gwat_q2_model_null, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.5, 3.0)
    plt.ylim(15, 26)
    plt.ylabel('Area$_{Q2}$ ($\cdot 10^3 \mu m^2$)', fontsize=14)

    plt.subplot(324)
    cytometer.stats.plot_linear_regression(iwat_q2_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q2', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(iwat_q2_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q2', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(iwat_q2_model_null, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.2, 2.1)
    plt.ylim(15, 26)

    plt.subplot(325)
    cytometer.stats.plot_linear_regression(gwat_q3_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q3', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(gwat_q3_model, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'gWAT'}, sy=1e-3,
                                           dep_var='area_Q3', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(gwat_q3_model_null, df_slides[df_slides['depot'] == 'gWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.5, 3.0)
    plt.ylim(30, 45)
    plt.xlabel('Depot weight (g)', fontsize=14)
    plt.ylabel('Area$_{Q3}$ ($\cdot 10^3 \mu m^2$)', fontsize=14)

    plt.subplot(326)
    cytometer.stats.plot_linear_regression(iwat_q3_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:WT', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q3', c='C0', marker='x',
                                           line_label='WT')
    cytometer.stats.plot_linear_regression(iwat_q3_model, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           other_vars={'Genotype': 'Arl15-Del2:Het', 'depot': 'iWAT'}, sy=1e-3,
                                           dep_var='area_Q3', c='C1', marker='o',
                                           line_label='Het')
    cytometer.stats.plot_linear_regression(iwat_q3_model_null, df_slides[df_slides['depot'] == 'iWAT'], ind_var='DW',
                                           sy=1e-3, c='k--', line_label='All')
    plt.tick_params(labelsize=14)
    plt.xlim(1.2, 2.1)
    plt.ylim(30, 45)
    plt.xlabel('Depot weight (g)', fontsize=14)

    plt.tight_layout()

    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_area_quartile_models.png'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_area_quartile_models.jpg'))
    plt.savefig(os.path.join(figures_dir, 'arl15del2_exp_0003_paper_figures_area_quartile_models.svg'))
