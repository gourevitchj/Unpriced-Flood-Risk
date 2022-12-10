### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: figures.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os
import time
import pickle
import matplotlib
import subprocess
import numpy as np
import pandas as pd
import seaborn as sb
import geopandas as gpd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import paths, reference, utils

### Suppress Pandas warnings
pd.options.mode.chained_assignment = None

### Suppress warnings
import warnings
warnings.filterwarnings('ignore')


def empiricaldiscount_combinedgroups():

	"""Generate point plots for flood zone discounts.

	Arguments:
		None

	Returns:
		None
	"""
	
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(12/1.3, 7/1.3))

	### Initialize path to CSV with summary of hedonic model outputs
	outputs_csv_uri = os.path.join(
		paths.outputs_dir, 'HedonicOutputs_Summary.csv')

	### Read CSV with summary of hedonic model outputs to Pandas DataFrame
	df = pd.read_csv(outputs_csv_uri)
	
	### Set formatting parameters
	scalar, fmt, lw, ms = 0.2, 'o-', 1.5, 8

	### Get national data
	model_label = 'attitude-x_coastal-x_disclosure-x'
	row = df[(df['model_label']==model_label) & (~df['100yr_coeff'].isna())]
	coeff = float(row['100yr_coeff'])
	se = float(row['100yr_se'])

	### Plot national data
	axes.axhline(coeff, color='k', linestyle='--', alpha=0.5)
	axes.fill_between([0.5, 2.5], y1=coeff-(2*se), y2=coeff+(2*se), 
					  color='k', ec='none', alpha=0.2)

	### Get county groups data
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	coeff_a = float(model_a['100yr_coeff'].dropna())
	coeff_b = float(model_b['100yr_coeff'].dropna())
	coeff_c = float(model_c['100yr_coeff'].dropna())
	coeff_d = float(model_d['100yr_coeff'].dropna())

	std_error_a = float(model_a['100yr_se'].dropna())
	std_error_b = float(model_b['100yr_se'].dropna())
	std_error_c = float(model_c['100yr_se'].dropna())
	std_error_d = float(model_d['100yr_se'].dropna())

	### Specify location on x-axis for no disclosure requirements
	x = 1

	### Plot coeff for no disclosure  & below median climate concern
	errorbar_dict = {'fmt':fmt, 'color':'purple', 'lw':lw, 'ms':ms, 'mfc':'w'}
	axes.errorbar(x-scalar, coeff_a, yerr=2*std_error_a, **errorbar_dict)

	### Plot coeff for no disclosure  & above median climate concern
	errorbar_dict = {'fmt':fmt, 'color':'purple', 'lw':lw, 'ms':ms}
	axes.errorbar(x+scalar, coeff_c, yerr=2*std_error_c, **errorbar_dict)

	### Specify location on x-axis for at least one disclosure 
	x = 2

	### Plot coeff for at least one disclosure  & below median climate concern
	errorbar_dict = {'fmt':fmt, 'color':'green', 'lw':lw, 'ms':ms, 'mfc':'w'}
	axes.errorbar(x-scalar, coeff_b, yerr=2*std_error_b, **errorbar_dict)
	
	### Plot coeff for at least one disclosure  & below above climate concern
	errorbar_dict = {'fmt':fmt, 'color':'green', 'lw':lw, 'ms':ms}
	axes.errorbar(x+scalar, coeff_d, yerr=2*std_error_d, **errorbar_dict)

	### Plot formatting
	axes.set_xlim(0.5, 2.5)
	axes.set_ylim(-0.13, 0)

	axes.set_ylabel('Empirical flood zone discount (%)')

	axes.set_xticks([1, 2])
	axes.set_xticklabels(
		['No disclosure requirements', 
		'At least one disclosure requirement'])

	ytick_labels = [round(t*100) for t in axes.get_yticks()]
	axes.set_yticklabels(ytick_labels)

	### Create legend labels
	axes.plot(-1, 0, 'ko-', ms=ms, mfc='w', label='Below median climate concern')
	axes.plot(-1, 0, 'ko-', ms=ms, label='Above median climate concern')

	### Create legend
	axes.legend(loc='lower left')

	### Save figure
	fn = 'empiricaldiscount_combinedgroups.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def empiricaldiscount_combinedgroups_xsmodel():

	"""Generate point plots for flood zone discounts for cross-sectional model.

	Arguments:
		None

	Returns:
		None
	"""
	
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(12/1.2, 7/1.2))

	### Initialize path to CSV with summary of  XS model outputs
	outputs_csv_uri = os.path.join(paths.xs_coeffs_csv_uri)

	### Read CSV with summary of XS model outputs to Pandas DataFrame
	df = pd.read_csv(outputs_csv_uri)

	### Drop coefficients for properties not at risk of flooding
	df = df[df['risk']!=0]
	
	### Set formatting parameters
	scalar, lw, ms = 0.25, 1.5, 8

	### Specify x for no disclosure requirements & below median climate concern
	x = 1
	df1 = df[df['group']=='ncnd']
	fmt_dict = {'ms':ms, 'mfc':'w', 'mec':'purple', 'ecolor': 'purple'}

	### Plot
	axes.errorbar(x-scalar, df1['coefficient'][df1['fz_risk']=='100'], 
		2*df1['se'][df1['fz_risk']=='100'], marker='o', **fmt_dict)
	
	axes.errorbar(x, df1['coefficient'][df1['fz_risk']=='500'], 
		2*df1['se'][df1['fz_risk']=='500'], marker='s', **fmt_dict)
	
	axes.errorbar(x+scalar, df1['coefficient'][df1['fz_risk']=='outside'], 
		2*df1['se'][df1['fz_risk']=='outside'], marker='^', **fmt_dict)


	### Specify x for no disclosure requirements & above median climate concern
	x = 2
	df2 = df[df['group']=='cnd']
	fmt_dict = {'ms':ms, 'mfc':'purple', 'mec':'purple', 'ecolor': 'purple'}

	### Plot
	axes.errorbar(x-scalar, df2['coefficient'][df2['fz_risk']=='100'], 
		2*df2['se'][df2['fz_risk']=='100'], marker='o', **fmt_dict)

	axes.errorbar(x, df2['coefficient'][df2['fz_risk']=='500'], 
		2*df2['se'][df2['fz_risk']=='500'], marker='s',  **fmt_dict)

	axes.errorbar(x+scalar, df2['coefficient'][df2['fz_risk']=='outside'], 
		2*df2['se'][df2['fz_risk']=='outside'], marker='^',  **fmt_dict)

	### Specify x for at least one disclosure & below median climate concern
	x = 3
	df3 = df[df['group']=='ncd']
	fmt_dict = {'ms':ms, 'mfc':'w', 'mec':'green', 'ecolor': 'green'}

	### Plot
	axes.errorbar(x-scalar, df3['coefficient'][df3['fz_risk']=='100'], 
		df3['se'][df3['fz_risk']=='100'], marker='o',  **fmt_dict)

	axes.errorbar(x, df3['coefficient'][df3['fz_risk']=='500'], 
		df3['se'][df3['fz_risk']=='500'], marker='s',  **fmt_dict)

	axes.errorbar(x+scalar, df3['coefficient'][df3['fz_risk']=='outside'], 
		df3['se'][df3['fz_risk']=='outside'], marker='^',  **fmt_dict)

	### Specify x for at least one disclosure & above median climate concern
	x = 4
	df4 = df[df['group']=='cd']
	fmt_dict = {'ms':ms, 'mfc':'green', 'mec':'green', 'ecolor': 'green'}

	### Plot
	axes.errorbar(x-scalar, df4['coefficient'][df4['fz_risk']=='100'], 
		df4['se'][df4['fz_risk']=='100'], marker='o',  **fmt_dict)

	axes.errorbar(x, df3['coefficient'][df3['fz_risk']=='500'], 
		df3['se'][df3['fz_risk']=='500'], marker='s',  **fmt_dict)

	axes.errorbar(x+scalar, df4['coefficient'][df4['fz_risk']=='outside'], 
		df4['se'][df4['fz_risk']=='outside'], marker='^',  **fmt_dict)

	## Plot formatting
	axes.set_xlim(0.5, 4.5)

	axes.set_ylabel('Empirical flood zone discount (%)')

	axes.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
	axes.set_xticklabels([
		'',
		'\n\nNo disclosure requirements', 
		'',
		'\n\nAt least one disclosure requirement', 
		''],
		weight = 'bold')

	text_fmt = {'ha':'center', 'fontstyle': 'italic'}
	y_loc = axes.get_ylim()[0] - ((axes.get_ylim()[1] - axes.get_ylim()[0]) * 0.08)
	axes.text(1, y_loc, 'Below median\nclimate concern', **text_fmt)
	axes.text(2, y_loc, 'Above median\nclimate concern', **text_fmt)
	axes.text(3, y_loc, 'Below median\nclimate concern', **text_fmt)
	axes.text(4, y_loc, 'Above median\nclimate concern', **text_fmt)

	for x in [1.5, 2.5, 3.5]:
		axes.axvline(x=x, color='k', linestyle='--', alpha=0.5)

	axes.axhline(y=0, color='k')

	ytick_labels = [round(t*100) for t in axes.get_yticks()]
	axes.set_yticklabels(ytick_labels)

	### Create legend labels
	axes.plot(-1, 0, 'ko', ms=ms, label='100-year flood zone')
	axes.plot(-1, 0, 'ks', ms=ms, label='500-year flood zone')
	axes.plot(-1, 0, 'k^', ms=ms, label='Outside flood zone')

	### Create legend
	axes.legend(loc='lower left', fontsize=10)

	### Save figure
	fn = 'empiricaldiscount_combinedgroups_xsmodel.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def overvaluation_maps(discount_rate=3, hazard_scenario='m'):

	"""Generate overvaluation maps.

	Arguments:
		None

	Optional arguments:
		discount_rate (int): Applied discount rate as an integer 
		hazard_scenario (string): Low ('l'), medium ('m'), or high 
			('h') hazard scenario

	Returns:
		None
	"""
		
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 8))
	fig.subplots_adjust(hspace=0.1, wspace=0.3)

	### Read outputs summary to Pandas dataframe
	outputs_csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')
	df = pd.read_csv(outputs_csv_uri)

	### Initialize paths to shapefiles
	states_shp_uri = paths.states_shp_uri
	counties_shp_uri = paths.counties_shp_uri

	### Read shapefiles to GeoPandas dataframes
	states_df = gpd.read_file(states_shp_uri)
	counties_df = gpd.read_file(counties_shp_uri)

	counties_df['fips'] = counties_df['GEOID'].astype(int)
	counties_df = counties_df[['fips', 'geometry']]

	### Merge df and counties_df 
	df = counties_df.merge(df, on='fips', how='right')

	### Subset data to 3% discount rate
	df = df[df['discount_rate']==discount_rate]

	### Subset data to EAL method 
	eal_method = 'fld_eal_base_noFR_mid_fs_%s' %hazard_scenario
	df = df[df['eal_method']==eal_method]

	### Assign axes
	ax_topleft  = axes[0,0]
	ax_bottomleft = axes[1,0]
	ax_topright = axes[0,1]
	ax_bottomright = axes[1,1]

	axes_list = [ax_topleft, ax_topright, ax_bottomleft, ax_bottomright]

	### Set equal aspect
	for ax in axes_list:
		ax.set_aspect('equal')

	### Populate legend properties
	def create_legend(ax, bins, cmap):
		legend_dict = {}
		legend_dict['legend'] = True
		divider = make_axes_locatable(ax)
		cax = divider.append_axes('right', size='5%', pad=0)	
		cax.yaxis.set_label_position('right')
		legend_dict['cax'] = cax
		legend_dict['cmap'] = cmap
		legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
				boundaries=bins, ncolors=len(bins)-1)

		return legend_dict

	##################### Plot data for top-left plot (A) ######################
	ax = ax_topleft

	### Read county groups CSV to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Get model outputs
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	coeff_100yr_group_a = float(model_a['100yr_coeff'].dropna())
	coeff_100yr_group_b = float(model_b['100yr_coeff'].dropna())
	coeff_100yr_group_c = float(model_c['100yr_coeff'].dropna())
	coeff_100yr_group_d = float(model_d['100yr_coeff'].dropna())

	se_100yr_group_a = float(model_a['100yr_se'].dropna())
	se_100yr_group_b = float(model_b['100yr_se'].dropna())
	se_100yr_group_c = float(model_c['100yr_se'].dropna())
	se_100yr_group_d = float(model_d['100yr_se'].dropna())

	### Create 'coeff' column in data frame 
	col_name = 'coeff'
	counties_df[col_name] = 0.5

	attitude_col = 'perc_personal_binary'
	disclosure_col = 'disclosure_binary'

	group_a_fips = df_groups['fips'][(df_groups[attitude_col]==0) & 
									 (df_groups[disclosure_col]==0)] 
	group_b_fips = df_groups['fips'][(df_groups[attitude_col]==0) & 
									 (df_groups[disclosure_col]==1)] 
	group_c_fips = df_groups['fips'][(df_groups[attitude_col]==1) & 
									 (df_groups[disclosure_col]==0)] 
	group_d_fips = df_groups['fips'][(df_groups[attitude_col]==1) & 
									 (df_groups[disclosure_col]==1)] 

	counties_df[col_name] = np.where(
		counties_df['fips'].isin(group_a_fips), 3.5, counties_df[col_name])
	counties_df[col_name] = np.where(
		counties_df['fips'].isin(group_b_fips), 2.5, counties_df[col_name])
	counties_df[col_name] = np.where(
		counties_df['fips'].isin(group_c_fips), 1.5, counties_df[col_name])
	counties_df[col_name] = np.where(
		counties_df['fips'].isin(group_d_fips), 0.5, counties_df[col_name])								 									 

	### Get bins
	bins = range(5)

	### Get colormap
	cmap = plt.get_cmap('plasma', 10)
	colors = [cmap(1), cmap(3), cmap(6), cmap(8)]
	cmap = ListedColormap(colors, N=256)

	### Create legend
	legend_dict = create_legend(ax, bins, cmap)

	### Plot data
	counties_df.plot(column=col_name, antialiased=False, ec='none', 
			ax=ax, zorder=2, **legend_dict)

	### Set colorbar ticks
	ticks = [0.5, 1.5, 2.5, 3.5]
	legend_dict['cax'].yaxis.set_ticks(ticks)

	### Set colorbar tick labels $\bf{{a}}$
	labels = [ 
		r'$\bf{Group\ A}$'+'\n' +
		'No disclosure requirement\n' +
		'Below median climate concern\n' +
		f'Estimated SFHA discount: {coeff_100yr_group_a*100:.1f}%\n'+
		f'95% CI: {(coeff_100yr_group_a - 2*se_100yr_group_a)*100:.1f}% to ' +
				 f'{(coeff_100yr_group_a + 2*se_100yr_group_a)*100:.1f}%',

		r'$\bf{Group\ B}$'+'\n' +
		'At least one requirement\n' +
		'Below median climate concern\n' + 
		f'SFHA discount: {coeff_100yr_group_b*100:.1f}%\n' +
		f'95% CI: {(coeff_100yr_group_b - 2*se_100yr_group_b)*100:.1f}% to ' +
				 f'{(coeff_100yr_group_b + 2*se_100yr_group_b)*100:.1f}%',
		
		r'$\bf{Group\ C}$'+'\n'+
		'No disclosure requirement\n' +
		'Above median climate concern\n' +
		f'SFHA discount: {coeff_100yr_group_c*100:.1f}%\n' +
		f'95% CI: {(coeff_100yr_group_c - 2*se_100yr_group_c)*100:.1f}% to '+
				 f'{(coeff_100yr_group_c + 2*se_100yr_group_c)*100:.1f}%',
		
		r'$\bf{Group\ D}$'+'\n'+
		'At least one requirement\n' +
		'Above median climate concern\n' +
		f'SFHA discount: {coeff_100yr_group_d*100:.1f}%\n' +
		f'95% CI: {(coeff_100yr_group_d - 2*se_100yr_group_d)*100:.1f}% to ' +
				 f'{(coeff_100yr_group_d + 2*se_100yr_group_d)*100:.1f}%',
		][::-1]

	legend_dict['cax'].set_yticklabels(labels, fontsize=8)

	legend_dict['cax'].tick_params(axis='y', right=False, pad=0) 


	#################### Plot data for top-right plot (B) ######################
	ax = ax_topright

	col_name = 'percovervalued_npv>0_median'
	bins = [0, 1, 2, 3, 4, 5, 10, 20, 30, 50]
	cmap = plt.get_cmap('BuPu', len(bins)-1)
	legend_dict = create_legend(ax, bins, cmap)
	df[col_name] = df[col_name] * 100
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=ax, zorder=2, **legend_dict)

	### Set colorbar tick labels
	legend_dict['cax'].set_yticklabels(['%d' %b + '%' for b in bins])


	################### Plot data for bottom-left plot (C) #####################
	ax = ax_bottomleft
	
	col_name = 'overvaluation_percentoftotal'
	bins = [0, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 25]
	cmap = plt.get_cmap('PuBuGn', len(bins)-1)
	legend_dict = create_legend(ax, bins, cmap)
	
	df[col_name] = (df['monetary_overvaluation_sum'] / df['fmv_sum']) * 100
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=ax, zorder=2, **legend_dict)
	
	### Set colorbar tick labels
	legend_dict['cax'].set_yticklabels(['%s' %b + '%' for b in bins])


	################### Plot data for bottom-right plot (D) ####################
	ax = ax_bottomright

	col_name = 'monetary_overvaluation_sum'
	bins = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
	cmap = plt.get_cmap('YlOrBr', len(bins)-1)
	legend_dict = create_legend(ax, bins, cmap)
	df[col_name] = df[col_name] / 10**6
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=ax, zorder=2, **legend_dict)
	
	### Set colorbar tick labels
	legend_dict['cax'].set_yticklabels([
		'$0', '$1 million', '$5 million', '$10 million', 
		'$50 million', '$100 million', '$500 million', 
		'$1 billion', '$5 billion', '$10 billion'])

	############################################################################

	### Plot state boundaries
	for ax in axes_list:	
		states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=ax, zorder=1)
		states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=ax, zorder=1)

		states_df.plot(ec='k', fc='none', lw=0.4, ax=ax, zorder=3)
		states_df.plot(ec='k', fc='none', lw=0.4, ax=ax, zorder=3)

	### Plot formatting
	for ax, letter in zip(axes_list, ['A','B','C','D']):
		ax.set_xticks([])
		ax.set_yticks([])

		ax.annotate(letter, xy=(0.05,0.95), xycoords='axes fraction',
					fontsize=12, fontweight='bold')

		### Hide spines
		for j in ['left', 'right', 'top', 'bottom']:
			ax.spines[j].set_visible(False)

	### Save figure
	fn = 'overvaluation_maps_dr%d_%s.png' %(discount_rate, hazard_scenario)
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def overvaluation_rankedbystate():

	"""Generate figure for overvaluation ranked by state.

	Arguments:
		None

	Returns:
		None
	"""

	### Initialize path to CSV
	csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')	

	### Read CSV to Pandas dataframe
	df = pd.read_csv(csv_uri)

	### Prep data
	df['state_fips'] = df['fips'].astype(str).str.slice(start=0, stop=-3)
	df['state_fips'] = np.where(df['state_fips'].str.len()==1, 
	 	'0'+df['state_fips'], df['state_fips'])

	df = df[df['eal_method']=='fld_eal_base_noFR_mid_fs_m']

	df = df.groupby(['state_fips', 'discount_rate'], as_index=False).sum()

	df = df[['state_fips', 'discount_rate', 'monetary_overvaluation_sum']]
	df['monetary_overvaluation_sum'] /= 10**9


	df['state_name'] = df['state_fips'].map(reference.state_fips_dict)

	df_1 = df[df['discount_rate']==1]
	df_3 = df[df['discount_rate']==3]
	df_5 = df[df['discount_rate']==5]
	df_7 = df[df['discount_rate']==7]

	df_1 = df_1.sort_values('monetary_overvaluation_sum', ascending=False)
	df_3 = df_3.sort_values('monetary_overvaluation_sum', ascending=False)
	df_5 = df_5.sort_values('monetary_overvaluation_sum', ascending=False)
	df_7 = df_7.sort_values('monetary_overvaluation_sum', ascending=False)

	df_1 = df_1.reset_index(drop=True)
	df_3 = df_3.reset_index(drop=True)
	df_5 = df_5.reset_index(drop=True)
	df_7 = df_7.reset_index(drop=True)

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 3.5))
	fig.subplots_adjust(hspace=0, wspace=0.3)

	### Plot data
	width = 0.8
	cmap = plt.get_cmap('viridis', 4)
	for x, state in enumerate(df_1['state_fips']): 
		y_1 = df_1['monetary_overvaluation_sum'][df_1['state_fips']==state]
		axes.bar(x, y_1, width, 
			label='1%', zorder=1, color=cmap(0), clip_on=True, alpha=0.9)

		y_3 = df_3['monetary_overvaluation_sum'][df_3['state_fips']==state]
		axes.bar(x, y_3, width, 
			label='3%', zorder=2, color=cmap(1), clip_on=True, alpha=0.9)

		y_5 = df_5['monetary_overvaluation_sum'][df_5['state_fips']==state]
		axes.bar(x, y_5, width, 
			label='5%', zorder=3, color=cmap(2), clip_on=True, alpha=0.9)

		y_7 = df_7['monetary_overvaluation_sum'][df_7['state_fips']==state]
		axes.bar(x, y_7, width, 
			label='7%', zorder=4, color=cmap(3), clip_on=True, alpha=0.9)

		print (state, y_3)

	### Format plot
	axes.set_xticks(range(len(df_1['state_name'].unique())))
	axes.set_xticklabels(df_1['state_name'], rotation=90)
	axes.set_ylabel('Overvaluation (billions $)')

	### Create legend
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), title='Discount Rate')

	### Save figure
	fn = 'overvaluation_rankedbystate.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def overvaluation_bypopulationgroups_v2(
	assume_nonsfha_discount=False, assume_xs_discount=False):

	"""Generate figure for overvaluation by population groups.

	Arguments:
		None 
	
	Option arguments:
		assume_nonsfha_discount (boolean): If True, assume non-SFHA properties
			exposed to flood risk are discounted at the same rate as SFHA properties 
		assume_xs_discount (boolean): If True, use flood zone discounts estimated
			by the cross-sectional model
			
	Returns:
		None
	"""

	### Get ACS data
	df1 = pd.read_csv(os.path.join(paths.outputs_dir, 'CensusTract_Outputs.csv'))
	df2 = pd.read_csv(paths.acs_csv_uri)

	### Get overvaluation outputs
	if assume_nonsfha_discount == True:
		df1 = pd.read_csv(os.path.join(paths.outputs_dir, 
			'CensusTract_Outputs_AssumeNonSFHAdiscount.csv'))

	if assume_xs_discount == True:
		df1 = pd.read_csv(os.path.join(paths.outputs_dir, 
			'CensusTract_Outputs_AssumeXSdiscount.csv'))

	### Prep df1
	df1 = df1[df1['tract']!='000nan']

	df1['tract'] = df1['tract'].astype(float).astype(int).astype(str)

	df1['tract'] = np.where(df1['tract'].str.len()==3, 
		'000' + df1['tract'], df1['tract'])

	df1['tract'] = np.where(df1['tract'].str.len()==4, 
		'00' + df1['tract'], df1['tract'])

	df1['tract'] = np.where(df1['tract'].str.len()==5, 
		'0' + df1['tract'], df1['tract'])

	df1['geoid'] = df1['county'].astype(str) + df1['tract'] 

	### Prep df2
	df2['GEOID'] = df2['GEOID'].astype(int).astype(str)

	### Merge data
	df = df1.merge(df2, how='left', left_on='geoid', right_on='GEOID')

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,11), 
		gridspec_kw={'width_ratios': [5.5, 1]})
	fig.subplots_adjust(hspace=0.15, wspace=0.05)

	### Initialize overvaluation bins
	ov_bins = [0, 5, 10, 25, 50, 100]

	### Initialize discount rate
	dr = 3

	### Annotate plots
	for i, letter in enumerate(['A', 'B']):
		axes[i,0].annotate(letter, xy=(-0.04, 1.04), xycoords='axes fraction',
					fontsize=12, fontweight='bold', annotation_clip=False)


	##################### Plot distribution by income groups ###################

	ax = axes[0,0]

	### Initialize income percentile bins
	income_bins = [0, 20, 40, 60, 80, 100]

	### Initialize colormap
	cmap = plt.get_cmap('PuBu', len(income_bins))

	ax.vlines(0.5, 0, 15, 'k', alpha=0.7, linestyle='--')
	ax.vlines(1.5, 0, 15, 'k', alpha=0.7, linestyle='--')
	ax.vlines(2.5, 0, 15, 'k', alpha=0.7, linestyle='--')
	ax.vlines(3.5, 0, 15, 'k', alpha=0.7, linestyle='--')

	### Plot data
	for i in range(len(ov_bins)-1):
		ov_l = ov_bins[i] 
		ov_u = ov_bins[i+1]

		df['n_properties_binsum'] = [0] * len(df)
		for b in range(ov_l, ov_u, 5):
			col = 'n_properties_ov%dto%d_%d' %(b, b+5, dr)
			df['n_properties_binsum'] += df[col]

		### Iterate through deciles
		for j in range(len(income_bins)-1):
			inc_l = income_bins[j]
			inc_u = income_bins[j+1]

			df_i = df[(df['B19013e1'] >  np.nanpercentile(df['B19013e1'], inc_l)) &
				  	  (df['B19013e1'] <= np.nanpercentile(df['B19013e1'], inc_u))] 

			x = (i - 0.5) + ((1 / len(income_bins)) * (j+1))
			y = (df_i['n_properties_binsum'].sum() / df_i['n_properties'].sum()) * 100

			w = (1 / len(income_bins)) * 0.8
			c = cmap(j+1)
			l = '(%d, %d]' %(income_bins[j], income_bins[j+1])

			ax.bar(x, y, width=w, color=c, label=l, ec='k')

			print(l, ov_l, y)

	### Plot formatting
	ax.set_xticks(range(len(ov_bins)-1))

	xtick_labels = []
	for l in range(len(ov_bins)-1):
		label = '%d - %d' %(ov_bins[l], ov_bins[l+1]) + '%'
		
		if '0 - 5%' in label:
			label = '>0 - 5%'

		xtick_labels.append(label)

	ax.set_xticklabels(xtick_labels)

	ax.set_ylabel('Percentage of properties (%)', 
		labelpad=10, fontweight='bold')
	
	ax.set_xlim(-0.5, 4.5)
	ax.set_ylim(0,10)

	ax.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    )

	### Create legend
	handles, labels = ax.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	legend = ax.legend(by_label.values(), by_label.keys(), 
		title='Household Income\nPercentiles', frameon=True)
	legend.get_frame().set_edgecolor('k')
	plt.setp(legend.get_title(), multialignment='center')

	print('')

	######################## Plot no overvaluation data ########################
	
	ax = axes[0,1]

	df['n_properties_binsum'] = [0] * len(df)
	for b in range(0, 100, 5):
		col = 'n_properties_ov%dto%d_%d' %(b, b+5, dr)
		df['n_properties_binsum'] += df[col]

	### Iterate through deciles
	for j in range(len(income_bins)-1):
		inc_l = income_bins[j]
		inc_u = income_bins[j+1]

		df_i = df[(df['B19013e1'] >  np.nanpercentile(df['B19013e1'], inc_l)) &
			  	  (df['B19013e1'] <= np.nanpercentile(df['B19013e1'], inc_u))] 

		x = (-0.5) + ((1 / len(income_bins)) * (j+1))
		y = ((df_i['n_properties_binsum'].sum()) / df_i['n_properties'].sum()) * 100
		y = 100 - y


		w = (1 / len(income_bins)) * 0.8
		c = cmap(j+1)
		l = '(%d, %d]' %(income_bins[j], income_bins[j+1])

		ax.bar(x, y, width=w, color=c, label=l, ec='k')

	ax.set_xticks([0])
	ax.set_xticklabels(['No Overvaluation'])
	ax.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    )
	ax.yaxis.tick_right()

	##################### Plot distribution by percent white ###################

	ax = axes[1,0]

	### Initialize income percentile bins
	percwhite_bins = [0, 20, 40, 60, 80, 100]

	### Get percent white data
	df['perc_white'] = df['B03002e3'] / df['B03002e1']

	### Initialize colormap
	cmap = plt.get_cmap('YlGn', len(percwhite_bins))

	ax.vlines(0.5, 0, 16, 'k', alpha=0.7, linestyle='--')
	ax.vlines(1.5, 0, 16, 'k', alpha=0.7, linestyle='--')
	ax.vlines(2.5, 0, 16, 'k', alpha=0.7, linestyle='--')
	ax.vlines(3.5, 0, 16, 'k', alpha=0.7, linestyle='--')

	### Plot data
	for i in range(len(ov_bins)-1):
		ov_l = ov_bins[i] 
		ov_u = ov_bins[i+1]

		df['n_properties_binsum'] = [0] * len(df)
		for b in range(ov_l, ov_u, 5):
			col = 'n_properties_ov%dto%d_%d' %(b, b+5, dr)
			df['n_properties_binsum'] += df[col]

		### Iterate through deciles
		for j in range(len(percwhite_bins)-1):
			inc_l = percwhite_bins[j]
			inc_u = percwhite_bins[j+1]

			df_i = df[(df['perc_white'] >  np.nanpercentile(df['perc_white'], inc_l)) &
				  	  (df['perc_white'] <= np.nanpercentile(df['perc_white'], inc_u))] 

			x = (i - 0.5) + ((1 / len(percwhite_bins)) * (j+1))
			y = (df_i['n_properties_binsum'].sum() / df_i['n_properties'].sum()) * 100

			w = (1 / len(percwhite_bins)) * 0.8
			c = cmap(j+1)
			l = '(%d, %d]' %(percwhite_bins[j], percwhite_bins[j+1])

			ax.bar(x, y, width=w, color=c, label=l, ec='k')

			print(l, ov_l, y)

	### Plot formatting
	ax.set_xticks(range(len(ov_bins)-1))

	xtick_labels = []
	for l in range(len(ov_bins)-1):
		label = '%d - %d' %(ov_bins[l], ov_bins[l+1]) + '%'
		
		if '0 - 5%' in label:
			label = '>0 - 5%'

		xtick_labels.append(label)

	ax.set_xticklabels(xtick_labels)
	ax.set_xlabel(r'Overvaluation (as % of property value)', 
		labelpad=10, fontweight='bold')

	ax.set_ylabel('Percentage of properties (%)', 
		labelpad=10, fontweight='bold')
	
	ax.set_xlim(-0.5, 4.5)
	ax.set_ylim(0,12)

	ax.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    )

	### Create legend
	handles, labels = ax.get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	legend = ax.legend(by_label.values(), by_label.keys(), 
		title='Percent White\nPercentiles', frameon=True)
	legend.get_frame().set_edgecolor('k')
	plt.setp(legend.get_title(), multialignment='center')

	######################## Plot no overvaluation data ########################
	ax = axes[1,1]

	df['n_properties_binsum'] = [0] * len(df)
	for b in range(0, 100, 5):
		col = 'n_properties_ov%dto%d_%d' %(b, b+5, dr)
		df['n_properties_binsum'] += df[col]

	### Iterate through deciles
	for j in range(len(income_bins)-1):
		inc_l = income_bins[j]
		inc_u = income_bins[j+1]

		df_i = df[(df['perc_white'] >  np.nanpercentile(df['perc_white'], inc_l)) &
			  	  (df['perc_white'] <= np.nanpercentile(df['perc_white'], inc_u))] 

		x = (-0.5) + ((1 / len(percwhite_bins)) * (j+1))
		y = (df_i['n_properties_binsum'].sum() / df_i['n_properties'].sum()) * 100
		y = 100 - y


		w = (1 / len(income_bins)) * 0.8
		c = cmap(j+1)
		l = '(%d, %d]' %(income_bins[j], income_bins[j+1])

		axes[1,1].bar(x, y, width=w, color=c, label=l, ec='k')

	ax.set_xticks([0])
	ax.set_xticklabels(['No Overvaluation'])
	ax.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    )
	ax.yaxis.tick_right()

	############################################################################

	### Save figure
	fn = 'overvaluation_bypopulationgroups_v2.png'
	
	if assume_nonsfha_discount == True:
		fn = 'overvaluation_bypopulationgroups_AssumeNonSFHAdiscount_v2.png'

	if assume_xs_discount == True:
		fn = 'overvaluation_bypopulationgroups_AssumeXSdiscount_v2.png'

	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def overvaluation_bypopulationgroups_scatter():

	"""Generate figure for overvaluation by population groups as *scatter plot*.

	Arguments:
		None 
			
	Returns:
		None
	"""

	### Get ACS data
	df1 = pd.read_csv(os.path.join(paths.outputs_dir, 'CensusTract_Outputs.csv'))
	df2 = pd.read_csv(paths.acs_csv_uri)

	### Prep df1
	df1 = df1[df1['tract']!='000nan']

	df1['tract'] = df1['tract'].astype(float).astype(int).astype(str)

	df1['tract'] = np.where(df1['tract'].str.len()==3, 
		'000' + df1['tract'], df1['tract'])

	df1['tract'] = np.where(df1['tract'].str.len()==4, 
		'00' + df1['tract'], df1['tract'])

	df1['tract'] = np.where(df1['tract'].str.len()==5, 
		'0' + df1['tract'], df1['tract'])

	df1['geoid'] = df1['county'].astype(str) + df1['tract'] 

	### Prep df2
	df2['GEOID'] = df2['GEOID'].astype(int).astype(str)

	### Merge data
	df = df1.merge(df2, how='left', left_on='geoid', right_on='GEOID')

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(ncols=2, nrows=6, figsize=(11,15))
	fig.subplots_adjust(wspace=0.2)

	### Initialize overvaluation bins
	ov_bins = [0, 5, 10, 25, 50, 100]

	### Initialize discount rate
	dr = 3

	####################### Plot overvaluation by ACS dat ######################

	### Iterate through ACS variables and subplots
	for i in range(2):

		### Iterate through overvaluation bins
		for j in range(6):

			### Set subplot
			ax = axes[j,i]

			### Set x variable
			if i==0:
				x_var = 'B19013e1'

			if i==1:
				df['perc_white'] = (df['B03002e3'] / df['B03002e1']) * 100
				x_var = 'perc_white'

			### Set y variable
			if j==0:
				df['ov_perc_0to5'] = (df['n_properties_ov0to5_3'] / 
									  df['n_properties']) * 100
				y_var = 'ov_perc_0to5'

			if j==1:
				df['ov_perc_5to10'] = (df['n_properties_ov5to10_3'] / 
									  df['n_properties']) * 100
				y_var = 'ov_perc_5to10'

			if j==2:
				df['ov_perc_10to25'] = ((df['n_properties_ov10to15_3'] +
										 df['n_properties_ov15to20_3'] +
										 df['n_properties_ov20to25_3']) / 
									  df['n_properties']) * 100
				y_var = 'ov_perc_10to25'

			if j==3:
				df['ov_perc_25to50'] = ((df['n_properties_ov25to30_3'] +
										 df['n_properties_ov30to35_3'] +
										 df['n_properties_ov35to40_3'] +
										 df['n_properties_ov40to45_3'] +
										 df['n_properties_ov45to50_3']) / 
									  df['n_properties']) * 100
				y_var = 'ov_perc_25to50'

			if j==4:
				df['ov_perc_50to100'] = ((df['n_properties_ov50to55_3'] +
										  df['n_properties_ov55to60_3'] +
										  df['n_properties_ov60to65_3'] +
										  df['n_properties_ov65to70_3'] +
										  df['n_properties_ov70to75_3'] +
										  df['n_properties_ov75to80_3'] +
										  df['n_properties_ov80to85_3'] +
										  df['n_properties_ov85to90_3'] +
										  df['n_properties_ov90to95_3'] +
										  df['n_properties_ov95to100_3']) / 
									  df['n_properties']) * 100
				y_var = 'ov_perc_50to100'

			if j==5:
				df['ov_perc_no_overvaluation'] = 100 - ((
								df['n_properties_ov5to10_3'] +
								df['n_properties_ov5to10_3'] +
								df['n_properties_ov10to15_3'] +
								df['n_properties_ov15to20_3'] +
								df['n_properties_ov20to25_3'] +
								df['n_properties_ov25to30_3'] +
								df['n_properties_ov30to35_3'] +
								df['n_properties_ov35to40_3'] +
								df['n_properties_ov40to45_3'] +
								df['n_properties_ov45to50_3'] +
								df['n_properties_ov50to55_3'] +
								df['n_properties_ov55to60_3'] +
								df['n_properties_ov60to65_3'] +
								df['n_properties_ov65to70_3'] +
								df['n_properties_ov70to75_3'] +
								df['n_properties_ov75to80_3'] +
								df['n_properties_ov80to85_3'] +
								df['n_properties_ov85to90_3'] +
								df['n_properties_ov90to95_3'] +
								df['n_properties_ov95to100_3']) / 
								
								df['n_properties']) * 100
				y_var = 'ov_perc_no_overvaluation'

			### Get x and y data
			df = df[((df[x_var]>=0) & (df[x_var]<300000) & 
					 (df[y_var]>0)  & (df[y_var]<=100))]
			x = df[x_var]
			y = df[y_var]

			### Plot data as binned-scatterplot
			df_est = utils.binscatter(x=x_var, y=y_var, data=df, ci=(3,3))
			ax.plot(df_est[x_var], df_est[y_var], 'ko')
			ax.errorbar(df_est[x_var], df_est[y_var], 
				yerr=df_est['ci'], ls='', lw=2, alpha=0.2)

			### Set axes tick labels and limits
			if i==0:
				ax.set_xticks([0, 50000, 100000, 150000])
				ax.set_xlim(0, 150000)
			if i==1:
				ax.set_xticks([0, 20, 40 ,60, 80, 100])
				ax.set_xlim(0, 100)
			

			if j==0:
				ax.set_ylim(0, 20)

			elif j>0 and j<5:
				ax.set_ylim(0, ax.get_ylim()[1])

			else:
				# ax.set_ylim(ax.get_ylim()[0], 100)
				ax.set_ylim(90, 100)


			### Plot regression line
			rp = sb.regplot(x=x, y=y, ax=ax, ci=95, scatter=False,
					lowess=False, color='r', truncate=False, label='label')

			### Fit linear regression model
			X = sm.add_constant(x)
			model = sm.WLS(y, X, weights=df['B03002e1'])
			fitted_model = model.fit()
			print(fitted_model.summary())

			### Set axes labels
			if i==0 and j<5:
				ax.set_ylabel(
					'Percentage of properties\novervalued by %d-%d'
					%(ov_bins[j], ov_bins[j+1]) + '%')

				if j==0:
					ax.set_ylabel('Percentage of properties\novervalued by >0-5%')

			elif i==0 and j==5:
				ax.set_ylabel(
					'Percentage of properties\nnot overvalued')
			else:
				ax.set_ylabel('')

			if i==0 and j==5: 
				ax.set_xlabel('Household Median Income ($)')
			elif i==1 and j==5:
				ax.set_xlabel('Percent Population White (%)')
			else:
				ax.set_xlabel('')
		
			if i==0:
				ax.set_xticklabels(['0k', '50k', '100k', '150k'])

			### Add p-values
			slope = fitted_model.params[1]
			p = fitted_model.pvalues[1]
			t1 = ax.text(0.05, 0.78, 'slope = {:.2g}'.format(slope)+'\np = {:.2g}'.format(p),
				fontsize=11, transform=ax.transAxes)

			t1.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))

	############################################################################

	### Save figure
	fn = 'overvaluation_bypopulationgroups_scatter.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	
	try:
		subprocess.run(['open', uri])

	except:
		from PIL import Image
		im = Image.open(uri)
		im.show()

	return None


def overvaluation_munifinance():

	"""Generate figure for overvaluation by municipal finance.

	Arguments:
		None 
			
	Returns:
		None
	"""
	
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(11/1.2, 6/1.2))
	fig.subplots_adjust(hspace=0, wspace=0.3)

	### Initialize path to municipal finance CSV file
	munifinance_fn = 'Fiscal_Vulnerability.csv'
	munifinance_csv_uri = os.path.join(paths.outputs_dir, munifinance_fn)

	### Read municipal finance CSV to Pandas DataFrame
	df = pd.read_csv(munifinance_csv_uri)

	############################## Create colormap #############################

	### Define bi-variate colormap (https://github.com/rbjansen/xycmap)
	import xycmap
	corner_colors = ('white', 'orange', 'purple', 'black') #v3

	n = (10, 10)
	cmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)

	### Create discrete color bins
	xbins = []
	ybins = []

	for p in range(0,110,10):
		xbins.append(np.nanpercentile(df['sx'], p))
		ybins.append(np.nanpercentile(df['sy'], p))

	xbins = pd.Series(xbins)
	ybins = pd.Series(ybins)

	print(xbins)
	print('')
	print(ybins)

	### Get colors for each county
	colors = utils.bivariate_color(
		sx=df['sx'], sy=df['sy'], cmap=cmap, 
		xbins=xbins, ybins=ybins)
	colors = colors.to_list()

	### If RGBA value is greater than 1, change it to 1
	for c, color in enumerate(colors):
		colors[c] = list(color)
		for v in range(4):
			if color[v] > 1:
				colors[c][v] = 1

	### Assign colors to dataframe
	df['colors'] = pd.Series(colors)

	### Create colorbar axes
	cax = fig.add_axes([0.75, 0.2, 0.6, 0.6])

	### Put bi-variate legend inside of colorbar axes
	xlabels = range(0,11)
	ylabels = range(0,11)
	cax = xycmap.bivariate_legend(
		ax=cax, sx=df['sx'], sy=df['sy'], cmap=cmap, 
		xlabels=xlabels, ylabels=ylabels)

	cax.set_xticks([])
	cax.set_yticks([])

	### Set colormap axis labels
	cax.set_xlabel('Property Tax Revenue\n(as % of total revenue)', 
		labelpad=10, fontweight='bold')
	cax.set_ylabel('Property overvaluation\n(as % of total property value)', 
		labelpad=10, fontweight='bold')

	################################# Plot data ################################

	### Initialize paths to shapefiles
	states_shp_uri = paths.states_shp_uri
	counties_shp_uri = paths.counties_shp_uri

	### Read shapefiles to GeoPandas dataframes
	states_df = gpd.read_file(states_shp_uri)
	counties_df = gpd.read_file(counties_shp_uri)

	counties_df['fips'] = counties_df['GEOID'].astype(int)
	counties_df = counties_df[['fips', 'geometry']]

	### Merge df and counties_df 
	df = counties_df.merge(df, on='fips', how='inner')

	### Set equal aspect
	axes.set_aspect('equal')

	### Plot county data
	df.plot(ax=axes, ec=df['colors'], lw=0.2, fc=df['colors'], zorder=2)


	df2 = df[(df['sx']>=np.nanpercentile(df['sx'], 80)) & 
			 (df['sy']>=np.nanpercentile(df['sy'], 80))]
	
	df2.plot(ax=axes, ec='red', lw=0.8, fc=df2['colors'], zorder=4)

	### Plot state boundaries
	states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=axes, zorder=1)
	states_df.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Hide x and y ticks
	axes.set_xticks([])
	axes.set_yticks([])
	
	### Hide spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)

	### Save figure
	fn = 'overvaluation_munifinance.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def yaleclimatesurvey_map(): 

	"""Generate Yale Climate Survey map.

	Arguments:
		None 
			
	Returns:
		None
	"""
		
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 8))
	fig.subplots_adjust(hspace=0, wspace=0.1)

	### Read Yale Climate Survey CSV file to Pandas Dataframe
	df = pd.read_csv(paths.climatesurvey_csv_uri)
	
	### Read counties shapefile to GeoPandas dataframe
	df_counties = gpd.read_file(paths.counties_shp_uri)
	df_counties['GEOID'] = df_counties['GEOID'].astype(int)

	### Merge dataframes
	df = df_counties.merge(df, on='GEOID', how='right')

	### Populate legend properties
	legend_dict = {}
	legend_dict['legend'] = True
	divider = make_axes_locatable(axes)
	cax = divider.append_axes('right', size='5%', pad=0)	
	cax.yaxis.set_label_position('right')
	legend_dict['cax'] = cax

	bins = [np.percentile(df['personal'], i) for i in range(0, 110, 10)]
	cmap = plt.get_cmap('PuOr', len(bins)-1)

	legend_dict['cmap'] = cmap
	legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
			boundaries=bins, ncolors=len(bins)-1)

	### Plot county data
	df.plot(column='personal', antialiased=False, 
		lw=0.0, zorder=1, ax=axes, **legend_dict)

	### Plot states
	df_states = gpd.read_file(paths.states_shp_uri)
	df_states.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Hide ticks
	axes.set_xticks([])
	axes.set_yticks([])

	### Don't show spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)
		axes.spines[j].set_visible(False)

	### Set legend axis formatting
	cax.yaxis.set_major_locator(ticker.LinearLocator(numticks=11))
	cax.set_yticklabels(range(0, 110, 10))

	### Save figure
	fn = 'yale_map.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def firmupdates_map(): 

	"""Generate year of FIRM updates map.

	Arguments:
		None 
			
	Returns:
		None
	"""
	
	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 8))
	fig.subplots_adjust(hspace=0, wspace=0.1)

	### Read FIRM update CSV file to Pandas Dataframe
	df = pd.read_csv(paths.firm_effdate_csv_uri)
	df['date'] = pd.to_datetime(df['Effective_date_first_DFIRM'])
	df['year'] = pd. DatetimeIndex(df['date']).year

	df['year'] = np.where(df['year']>=2015, 2015, df['year'])
	df['year'] = np.where(df['year']<=2000, 2000, df['year'])

	### Read census tracts shapefile to GeoPandas dataframe
	df_tracts = gpd.read_file(paths.tracts_shp_uri)
	df_tracts['GEOID'] = df_tracts['GEOID'].astype(int)

	### Merge dataframes
	df = df_tracts.merge(df, on='GEOID', how='right')

	### Populate legend properties
	legend_dict = {}
	legend_dict['legend'] = True
	divider = make_axes_locatable(axes)
	cax = divider.append_axes('right', size='5%', pad=0)	
	cax.yaxis.set_label_position('right')
	legend_dict['cax'] = cax

	bins = range(2000, 2015)
	cmap = plt.get_cmap('magma_r', len(bins)-1)

	legend_dict['cmap'] = cmap
	legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
			boundaries=bins, ncolors=len(bins)-1)

	### Plot tract data
	df.plot(column='year', antialiased=False, 
		lw=0.0, zorder=2, ax=axes, **legend_dict)

	### Plot states
	df_states = gpd.read_file(paths.states_shp_uri)
	df_states.plot(ec='none', fc='lightgrey', lw=0.4, ax=axes, zorder=1)
	df_states.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Hide ticks
	axes.set_xticks([])
	axes.set_yticks([])

	### Don't show spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)
		axes.spines[j].set_visible(False)

	### Set legend axis formatting
	cax.set_yticklabels(['< 2000' , '2002', '2004', '2006', '2008', '2010',
						 '2012', '> 2014'])

	### Save figure
	fn = 'firmupdate_date.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def histograms_byfloodzone(
	assume_nonsfha_discount=False, assume_xs_discount=False):

	"""Generate histograms by flood zone.

	Arguments:
		None 
	
	Option arguments:
		assume_nonsfha_discount (boolean): If True, assume non-SFHA properties
			exposed to flood risk are discounted at the same rate as SFHA properties 
		assume_xs_discount (boolean): If True, use flood zone discounts estimated
			by the cross-sectional model	

	Returns:
		None
	"""

	### Initialize path to CSV file
	fn = 'Combined_AtRiskProperties.csv'

	if assume_nonsfha_discount == True:
		fn = 'Combined_AtRiskProperties_AssumeNonSFHAdiscount.csv'

	if assume_xs_discount == True:
		fn = 'Combined_AtRiskProperties_AssumeXSdiscount.csv'

	csv_uri = os.path.join(paths.outputs_dir, fn)

	### Read CSV file to Pandas DataFrame
	df = pd.read_csv(csv_uri)

	### Slice dataframe into SFHA groups
	df_sfha = df[df['flood_zone']=='100yr'].reset_index()
	df_nonsfha = df[df['flood_zone']!='100yr'].reset_index()

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 8))
	fig.subplots_adjust(hspace=0.2, wspace=0.2)

	### Set colors
	sfha_c = 'r'
	nonsfha_c = 'b' 

	################ Top-Left Plot: Percent Overvalued Histogram ###############

	### Set axes
	ax = axes[0,0]

	sfha_data = df_sfha['perc_overvalue_dr3'].dropna() * 100
	nonsfha_data = df_nonsfha['perc_overvalue_dr3'].dropna() * 100

	sfha_data = sfha_data[sfha_data>0]
	nonsfha_data = nonsfha_data[nonsfha_data>0]

	bins = np.arange(-10, 105, 0.5)

	### Plot data
	ax.hist([nonsfha_data, sfha_data], bins=bins, color=[nonsfha_c, sfha_c], 
		histtype='stepfilled', stacked=True, ec='k',
		label=['Non-SFHA Properties', 'SFHA Properties'])

	### Format axes
	ax.set_yscale('log')

	ax.set_xlim(0, 100)
	ax.set_ylim(ax.get_ylim()[0], 10**6)

	ax.set_xlabel('Overvaluation (%)')
	ax.set_ylabel('N Properties')

	ax.minorticks_off()

	### Legend
	ax.legend()

	############## Top-Right Plot: Dollars Overvalued Histogram ##############

	### Set axes
	ax = axes[0,1]

	sfha_data = df_sfha['dollars_overvalue_dr3'].dropna() / 10**6
	nonsfha_data = df_nonsfha['dollars_overvalue_dr3'].dropna() / 10**6

	sfha_data = sfha_data[(sfha_data>0) & (sfha_data<=4)]
	nonsfha_data = nonsfha_data[(nonsfha_data>0) & (nonsfha_data<=4)]

	bins = 200

	### Plot data
	ax.hist([nonsfha_data, sfha_data], 
		bins=bins, 
		color=[nonsfha_c, sfha_c], 
		histtype='stepfilled', stacked=True, ec='k',
		label=['Non-SFHA Properties', 'SFHA Properties'])

	### Format axes
	ax.set_yscale('log')

	ax.set_xlim(0, 4)
	ax.set_ylim(ax.get_ylim()[0], 10**6)

	ax.set_xticks(range(0,5))

	ax.set_xlabel('Overvaluation (Millions $)')
	ax.set_ylabel('N Properties')

	ax.minorticks_off()

	################# Bottom-Left Plot: Percent Overvalued CDF #################

	### Set axes
	ax = axes[1,0]

	x = np.arange(0, 1.0, 0.05)

	y_nonsfha = []
	y_sfha = []
	for i in x:
		df_i = df[(df['perc_overvalue_dr3']>i) & 
				  (df['perc_overvalue_dr3']<=i+0.05)]
		n = len(df_i)
		
		n_sfha = len(df_i[df_i['flood_zone']=='100yr'])
		perc_sfha = n_sfha / n
		y_sfha.append(perc_sfha)

		n_nonsfha = len(df_i[df_i['flood_zone']!='100yr'])
		perc_nonsfha = n_nonsfha / n
		y_nonsfha.append(perc_nonsfha)

	w = 0.05
	ax.bar(x+(w/2), y_nonsfha, w, 
		color=nonsfha_c, label='SFHA Properties')
	ax.bar(x+(w/2), y_sfha, w, bottom=y_nonsfha, 
		color=sfha_c, label='Non-SFHA Properties')

	### Format plot
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)

	ax.set_xticklabels([int(t*100) for t in ax.get_xticks()])

	ax.set_xlabel('Overvaluation (%)')
	ax.set_ylabel('Proportion SFHA / Non-SFHA')

	################# Bottom-Right Plot: Dollars Overvalued CDF ################

	### Set axes
	ax = axes[1,1]

	df = df[df['dollars_overvalue_dr3']>0]

	### Get data
	x = range(len(df))

	df = df.sort_values(by='dollars_overvalue_dr3', ascending=False).reset_index()
	df['dollars_overvalue_cum'] = df['dollars_overvalue_dr3'].cumsum()
	df['dollars_overvalue_cum'] /= 10**9
	y_col = 'dollars_overvalue_cum'

	ax.plot(x, df[y_col], 'k-')

	interval = 0.005
	for i in np.arange(0, 1.0, interval):
		x_lower = int(np.floor(i * len(df)))
		x_upper = int(np.floor((i+interval) * len(df)))

		x_subset = range(x_lower, x_upper)

		df_i = df[x_lower:x_upper].reset_index()

		y_subset = df_i[y_col]

		n_subset = len(df_i)
		n_subset_sfha = len(df_i[df_i['flood_zone']=='100yr'])
		n_subset_nonsfha = len(df_i[df_i['flood_zone']!='100yr'])
		perc_subset_sfha = n_subset_sfha / n_subset
		perc_subset_nonsfha = n_subset_nonsfha / n_subset

		y_subset_nonsfha = y_subset.max()*perc_subset_nonsfha
		y_subset_nonsfha = np.where(
			y_subset_nonsfha.max()>=y_subset, y_subset, y_subset_nonsfha)

		ax.fill_between(
			x_subset, 
			y1=y_subset_nonsfha, y2=y_subset, 
			ec=sfha_c, fc=sfha_c, lw=2)

		ax.fill_between(
			x_subset, 
			y_subset_nonsfha, 
			ec=nonsfha_c, fc=nonsfha_c, lw=2)

	### Format plot
	ax.set_xticklabels([int(t/10**6) for t in ax.get_xticks()])

	ax.set_xlabel('N Properties (Millions)')
	ax.set_ylabel('Cumulative Overvaluation (Billions $)')

	ax.set_xlim(-0.01 * ax.get_xlim()[1], len(df))
	ax.set_ylim(0, 200)

	############################################################################

	### Annotate subplots with letters
	for ax, letter in zip([axes[0,0], axes[0,1], axes[1,0], axes[1,1]], 
						  ['A','B','C','D']):

		ax.annotate(letter, xy=(-0.07, 1.04), xycoords='axes fraction',
					fontsize=12, fontweight='bold', annotation_clip=False)

	### Save figure
	fn = 'histograms_byfloodzone.png'

	if assume_nonsfha_discount == True:
		fn = 'histograms_byfloodzone_AssumeNonSFHAdiscount.png'

	if assume_xs_discount == True:
		fn = 'histograms_byfloodzone_AssumeXSdiscount.png'

	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def uncertainty_plots():

	"""Generate plots for uncertainty analysis.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Read CSV file to Pandas DataFrame
	csv_uri = os.path.join(paths.outputs_dir, 'MonteCarlo_Summary.csv')
	df = pd.read_csv(csv_uri)

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(nrows=3, figsize=(12, 6.5))
	fig.subplots_adjust(hspace=0.2, wspace=0.2)

	### Subset data
	df = df[(df['depthdamage_function']=='base') & 
			(df['damage_scenario']=='mid')
			]

	### Iterate through discount rates
	for i, dr in enumerate([7, 5, 3, 1]):
		c = plt.get_cmap('inferno_r', 6)(i+1)

		### Iterate through hazard scenarios
		for j, hs in enumerate(['l', 'm', 'h']):
		# for j, ds in enumerate(['low', 'mid', 'high']):
			ax = axes[j]

			data = df['overvaluation'][(df['discount_rate']==dr) & 
									   (df['hazard_scenario']==hs)
									   ]
									   
			### Convert data to billions
			data /= 10**9 

			mean = data.mean()
			std = data.std()

			print(dr, hs)
			print(mean)
			print(((mean+std - mean) / mean)*100)
			print('')

			### Plot data
			sb.kdeplot(data=data, ax=ax, color=c, alpha=0.4, 
					   bw_adjust=2, fill=True, label='%d%%' %dr)

			### Plot line for mean of data 
			ax.axvline(data.mean(), color=c, ls='--')

			### Annotate plot with scenario label
			if hs == 'l': hs_label = 'Low' 
			if hs == 'm': hs_label = 'Mid' 
			if hs == 'h': hs_label = 'High'
			ax.annotate('%s' %hs_label, fontweight='bold', 
				xy=(0.0, 0.05), xycoords='axes fraction')

			### Plot formatting
			ax.set_xticks(range(100,300,25))
			ax.set_xlim(100,275)

			# ax.set_ylim(0, 0.36)
			ax.set_yticks([])
			ax.set_ylabel('')


	### Formatting for top and middle plots
	for i in range(2):
		ax = axes[i]
		ax.set_xticks([])
		ax.set_xlabel('')

		for s in ['top',  'left', 'right']:
			ax.spines[s].set_visible(False)

	### Formatting for bottom plot
	ax = axes[2]
	for s in ['top', 'left', 'right']:
		ax.spines[s].set_visible(False)

	ax.set_xlabel('Overvaluation (Billion $)')

	### Create legend
	axes[0].legend(title='Discount Rate')

	### Save figure
	fn = 'uncertainty_plots.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def munifinance_demographics():

	"""Generate plots for municipal finance demographics.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(15,2.5), ncols=3)
	fig.subplots_adjust(wspace=0.2)

	### Initialize path to municipal finance CSV file
	munifinance_fn = 'Fiscal_Vulnerability.csv'
	munifinance_csv_uri = os.path.join(paths.outputs_dir, munifinance_fn)

	### Read municipal finance CSV to Pandas DataFrame
	df = pd.read_csv(munifinance_csv_uri)

	### Initialize path to county ACS CSV file
	county_acs_fn = 'acs_extract.csv'	
	county_acs_csv_uri = os.path.join(
		paths.data_dir, 'ACS/Counties', county_acs_fn)

	### Read county ACS CSV to Pandas DataFrame
	df_acs = pd.read_csv(county_acs_csv_uri)

	### Merge ACS dataframe to municipal finance dataframe
	df = df.merge(df_acs, how='left', left_on='fips', right_on='GEOID')

	df['perc_white'] = df['B03002e3'] / df['B03002e1']

	### Subset dataframe to only include vulnerable municipalities
	df_subset = df[(df['sx']>=np.nanpercentile(df['sx'], 80)) & 
				   (df['sy']>=np.nanpercentile(df['sy'], 80))]

	### Initialize general args
	args = {'stat':'density', 'alpha':0.4, 'fill':True}

	### Initialize colors
	args1 = args | {'color':'grey'}
	args2 = args | {'color':'r'}

	### Initialize number of bins
	bins = 50

	### Initialize column labels
	cols = ['B01003e1', 'B19013e1', 'perc_white']

	print(len(df_subset))

	### Iterate through column labels
	for i, col in enumerate(cols):
		### Set axis
		ax = axes[i]
		
		### If column is population size...
		if col=='B01003e1':
			ax.set_xlim(0,100000)
			ax.set_xlabel('Population size')

		### If column is household median income...
		if col=='B19013e1':
			ax.set_xlim(0,120000)
			ax.set_xlabel('Household median income ($)')

		### If column is percent white...
		if col=='perc_white':
			ax.set_xlim(0,1)
			ax.set_xlabel('Percent white (%)')

		df2 = df[(df[col]>=ax.get_xlim()[0]) & 
				 (df[col]<=ax.get_xlim()[1])]
		df_subset2 = df_subset[(df_subset[col]>=ax.get_xlim()[0]) & 
							   (df_subset[col]<=ax.get_xlim()[1])]

		df2 = df2[~df2['fips'].isin(df_subset2['fips'])]

		### Plot histograms
		sb.histplot(data=df2[col], ax=ax, bins=bins, 
			binrange=ax.get_xlim(), **args1)
		sb.histplot(data=df_subset2[col], ax=ax, bins=bins, 
			binrange=ax.get_xlim(), **args2)

		### Plot vertical lines for median values
		vaxline_args = {'linestyle':'--', 'zorder':10}
		ax.axvline(df2[col].median(), color='k', alpha=0.8, **vaxline_args)
		ax.axvline(df_subset2[col].median(), color='r', **vaxline_args)

		### Hide y-axis ticks, y-axis label, and top, left, and right spines
		ax.set_yticks([])
		ax.set_ylabel('')

		for s in ['top',  'left', 'right']:
			ax.spines[s].set_visible(False)

	### Save figure
	fn = 'munifinance_demographics.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def fmv_map():

	"""Generate map for median FMVs by county.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 8))

	### Read outputs summary to Pandas dataframe
	outputs_csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')
	df = pd.read_csv(outputs_csv_uri)

	### Initialize paths to shapefiles
	states_shp_uri = paths.states_shp_uri
	counties_shp_uri = paths.counties_shp_uri

	### Read shapefiles to GeoPandas dataframes
	states_df = gpd.read_file(states_shp_uri)
	counties_df = gpd.read_file(counties_shp_uri)

	counties_df['fips'] = counties_df['GEOID'].astype(int)
	counties_df = counties_df[['fips', 'geometry']]

	### Merge df and counties_df 
	df = counties_df.merge(df, on='fips', how='right')

	### Subset data to 3% discount rate
	df = df[df['discount_rate']==3]

	### Subset data to EAL method 
	df = df[df['eal_method']=='fld_eal_base_noFR_mid_fs_m']

	### Set equal aspect
	axes.set_aspect('equal')

	### Populate legend properties
	def create_legend(axes, bins, cmap):
		legend_dict = {}
		legend_dict['legend'] = True
		divider = make_axes_locatable(axes)
		cax = divider.append_axes('right', size='5%', pad=0)	
		cax.yaxis.set_label_position('right')
		legend_dict['cax'] = cax
		legend_dict['cmap'] = cmap
		legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
				boundaries=bins, ncolors=len(bins)-1)

		return legend_dict

	### Plot
	col_name = 'fmv_median'
	bins = list(range(0,600000,100000)) + [10**6]
	cmap = plt.get_cmap('YlGn', len(bins)-1)
	legend_dict = create_legend(axes, bins, cmap)
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=axes, zorder=2, **legend_dict)

	### Set colorbar tick labels
	legend_dict['cax'].set_yticklabels(["${:,.0f}".format(b) for b in bins])

	### Plot state boundaries
	states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=axes, zorder=1)
	states_df.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Plot formatting
	axes.set_xticks([])
	axes.set_yticks([])

	### Hide spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)

	### Save figure
	fn = 'fmv_map.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def ztrax_transactions_map():

	"""Generate map for number of transactions per parcel.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 8))
	fig.subplots_adjust(hspace=0, wspace=0.1)

	### Get list of files in cleaned ZTRAX directory
	fn_list = os.listdir(paths.ztrax_cleaned_csv_dir)

	### Create empty list to store paths to CSV files
	csv_uri_list = []

	### Iterate through file names in ZTRAX directory
	for fn in sorted(fn_list):

		### If suffix is '.csv'...
		if fn[-4:] == '.csv':

			### Get full CSV file path
			csv_uri = os.path.join(paths.ztrax_cleaned_csv_dir, fn)
			
			### Append full CSV file path to list
			csv_uri_list.append(csv_uri)

	### Initialize empty list to store outputs
	fips_dict = {}

	### Iterate through list of county CSV files
	for csv_uri in csv_uri_list:

		### Get county FIPS code 
		cnty_fips = str(int(os.path.basename(csv_uri)[:5]))

		### Print county FIPS code
		print('\t\t%s' %cnty_fips)

		### Read county CSV file to Pandas DataFrame
		df = pd.read_csv(csv_uri, low_memory=False, usecols=['pid', 'sid'])

		### Get number of transactions
		n_trans = len(df['sid'].unique())

		### Get number of properties
		n_properties = len(df['pid'].unique())

		### Get number of transactions per property
		trans_per_prop = float(n_trans) / float(n_properties)

		### Store number of transactions per property in fips_dct
		fips_dict[cnty_fips] = trans_per_prop

	with open('trans_per_prop.pickle', 'wb') as handle:
	    pickle.dump(fips_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	pickle_uri = os.path.join(paths.outputs_dir, 'trans_per_prop.pickle')
	with open(pickle_uri, 'rb') as handle:
	    fips_dict = pickle.load(handle)

	### Initialize paths to shapefiles
	states_shp_uri = paths.states_shp_uri
	counties_shp_uri = paths.counties_shp_uri

	### Read shapefiles to GeoPandas dataframes
	states_df = gpd.read_file(states_shp_uri)
	counties_df = gpd.read_file(counties_shp_uri)

	counties_df['fips'] = counties_df['GEOID'].astype(int)
	counties_df = counties_df[['fips', 'geometry']]

	### Map fips_dict onto counties dataframe
	df = counties_df.copy()
	df['trans_per_prop'] = df['fips'].astype(str).map(fips_dict)

	### Set equal aspect
	axes.set_aspect('equal')

	### Populate legend properties
	def create_legend(axes, bins, cmap):
		legend_dict = {}
		legend_dict['legend'] = True
		divider = make_axes_locatable(axes)
		cax = divider.append_axes('right', size='5%', pad=0)	
		cax.yaxis.set_label_position('right')
		legend_dict['cax'] = cax
		legend_dict['cmap'] = cmap
		legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
				boundaries=bins, ncolors=len(bins)-1)

		return legend_dict

	### Plot
	col_name = 'trans_per_prop'
	bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 3, 4]
	cmap = plt.get_cmap('PuBuGn', len(bins)-1)
	legend_dict = create_legend(axes, bins, cmap)
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=axes, zorder=2, **legend_dict)

	legend_dict['cax'].yaxis.set_major_locator(ticker.MaxNLocator(len(bins)))
	legend_dict['cax'].set_yticklabels(bins)

	### Plot state boundaries
	states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=axes, zorder=1)
	states_df.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Plot formatting
	axes.set_xticks([])
	axes.set_yticks([])
	
	### Hide spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)

	### Save figure
	fn = 'ztrax_transactions_map.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def npv_map():

	"""Generate map for for NPV of flood losses by county.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(figsize=(10, 8))

	### Read outputs summary to Pandas dataframe
	outputs_csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')
	df = pd.read_csv(outputs_csv_uri)

	### Initialize paths to shapefiles
	states_shp_uri = paths.states_shp_uri
	counties_shp_uri = paths.counties_shp_uri

	### Read shapefiles to GeoPandas dataframes
	states_df = gpd.read_file(states_shp_uri)
	counties_df = gpd.read_file(counties_shp_uri)

	counties_df['fips'] = counties_df['GEOID'].astype(int)
	counties_df = counties_df[['fips', 'geometry']]

	### Merge df and counties_df 
	df = counties_df.merge(df, on='fips', how='right')

	### Subset data to 3% discount rate
	df = df[df['discount_rate']==3]

	### Subset data to EAL method 
	df = df[df['eal_method']=='fld_eal_base_noFR_mid_fs_m']

	### Set equal aspect
	axes.set_aspect('equal')

	### Populate legend properties
	def create_legend(axes, bins, cmap):
		legend_dict = {}
		legend_dict['legend'] = True
		divider = make_axes_locatable(axes)
		cax = divider.append_axes('right', size='5%', pad=0)	
		cax.yaxis.set_label_position('right')
		legend_dict['cax'] = cax
		legend_dict['cmap'] = cmap
		legend_dict['norm'] = matplotlib.colors.BoundaryNorm(
				boundaries=bins, ncolors=len(bins)-1)

		return legend_dict

	### Plot
	col_name = 'npv_sum'
	bins = [0, 10**5, 5*10**5, 10**6, 5*10**6, 10**7, 5*10**7, 10**8, 5*10**8, 10**9]
	cmap = plt.get_cmap('RdPu', len(bins)-1)
	legend_dict = create_legend(axes, bins, cmap)
	df.plot(column=col_name, antialiased=False, ec='none', 
			ax=axes, zorder=2, **legend_dict)

	for p in range(0,110,10):
		print(np.percentile(df[col_name], p))

	### Set colorbar tick labels
	legend_dict['cax'].set_yticklabels(["${:,.0f}".format(b) for b in bins])

	### Plot state boundaries
	states_df.plot(ec='k', fc='lightgrey', lw=0.4, ax=axes, zorder=1)
	states_df.plot(ec='k', fc='none', lw=0.4, ax=axes, zorder=3)

	### Plot formatting
	axes.set_xticks([])
	axes.set_yticks([])

	### Hide spines
	for j in ['left', 'right', 'top', 'bottom']:
		axes.spines[j].set_visible(False)

	### Save figure
	fn = 'npv_map.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None


def npv_histograms():

	"""Generate histograms for NPV of flood losses by SFHA / non-SFHA.

	Arguments:
		None 
				
	Returns:
		None
	"""

	### Initialize path to CSV file
	fn = 'Combined_AtRiskProperties.csv'
	csv_uri = os.path.join(paths.outputs_dir, fn)

	### Set plot parameters and style
	sb.set(style='ticks')
	fig, axes = plt.subplots(ncols=2, figsize=(15/1.5, 6/1.5))

	### Read CSV file to Pandas DataFrame
	df = pd.read_csv(csv_uri)
	df = df[df['npv_dr3']>0]
	df['Flood Zone'] = np.where(df['flood_zone']=='100yr', 'SFHA', 'Non-SFHA')

	bins = 150

	### Plot data
	sb.histplot(df, x='npv_dr3', hue='Flood Zone', bins=bins, 
		alpha=0.8,
		binrange=[0,1*10**4],
		hue_order=['SFHA', 'Non-SFHA'],
		palette=['r', 'b'],
		ax=axes[0],
		legend=False
		)

	### Plot data
	sb.histplot(df, x='npv_dr3', hue='Flood Zone', bins=bins, 
		alpha=0.8,
		binrange=[0,5*10**5],
		hue_order=['SFHA', 'Non-SFHA'],
		palette=['r', 'b'],
		ax=axes[1],
		legend=True
		)

	sfha_median = df['npv_dr3'][df['Flood Zone']=='SFHA'].median()
	axes[0].axvline(sfha_median, color='r', ls='--')

	nonsfha_median = df['npv_dr3'][df['Flood Zone']=='Non-SFHA'].median()
	axes[0].axvline(nonsfha_median, color='b', ls='--')

	print(sfha_median)
	print(nonsfha_median)
	
	### Format axes
	axes[0].set_xlim(0, 1*10**4)
	axes[1].set_xlim(0, 5*10**5)
	
	for i in range(2):
		axes[i].set_yscale('log')
		axes[i].set_xlabel('NPV')
		axes[i].set_yticks([])
		axes[i].set_ylabel('')
		axes[i].set_xticklabels(
			["${:,.0f}".format(int(t)) for t in axes[i].get_xticks()])
		axes[i].tick_params(axis='y', which='minor', left=False)

		for s in ['top', 'left', 'right']:
			axes[i].spines[s].set_visible(False)

	### Save figure
	fn = 'npv_histograms.png'
	uri = os.path.join(paths.figures_dir, fn)
	plt.savefig(uri, bbox_inches='tight', dpi=600)
	plt.savefig(uri.replace('png', 'pdf'), bbox_inches='tight')

	### Open figure
	time.sleep(0.5)
	subprocess.run(['open', uri])

	return None

