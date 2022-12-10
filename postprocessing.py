### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: postprocessing.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os
import pickle
import warnings
import numpy as np
import pandas as pd

from scipy import stats
from joblib import Parallel, delayed

import paths, reference, utils

### Suppress performance warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


def collect_statsoutputs():

	"""Collect hedonic model outputs.

	Arguments:
		None
	
	Returns:
		None
	"""

	### Get list of pickle files
	pickle_dir = os.path.join(paths.outputs_dir, 'StatsModel_Pickles')
	fn_list = os.listdir(pickle_dir)
	pickle_uri_list = []
	for fn in fn_list:
		pickle_uri = os.path.join(pickle_dir, fn)
		pickle_uri_list.append(pickle_uri)

	### Summary rows list
	rows_list = []

	### Iterate through pickles
	for pickle_uri in sorted(pickle_uri_list):
		
		### Get model label
		model_label = os.path.basename(pickle_uri).split('.')[0]
		
		### Print model label
		print('\t\t%s' %model_label)

		### Read pickle file to dictionary
		model_dict = utils.read_pickle(pickle_uri)

		### Store outputs in row_dict
		row_dict = {}

		row_dict['model_label'] = model_dict['model_label']
		row_dict['n_trans'] = model_dict['n_trans']
		row_dict['n_props'] = model_dict['n_props']

		### If 100-yr dummy variable is in model, append model outputs
		if '100yr_coeff' in model_dict:		
			row_dict['100yr_coeff'] = model_dict['100yr_coeff']
			row_dict['100yr_se'] = model_dict['100yr_se']
			row_dict['100yr_ci_lower'] = model_dict['100yr_ci_lower']
			row_dict['100yr_ci_upper'] = model_dict['100yr_ci_upper']
			row_dict['100yr_pvalue'] = model_dict['100yr_pvalue']
			
		### If 500-yr dummy variable is in model, append model outputs
		if '500yr_coeff' in model_dict:
			row_dict['500yr_coeff'] = model_dict['500yr_coeff'] 
			row_dict['500yr_se'] = model_dict['500yr_se']
			row_dict['500yr_ci_lower'] = model_dict['500yr_ci_lower']
			row_dict['500yr_ci_upper'] = model_dict['500yr_ci_upper']
			row_dict['500yr_pvalue'] = model_dict['500yr_pvalue']

		### Append row_dict to rows_list
		rows_list.append(row_dict)

	### Convert list of dictionaries to Pandas DataFrame
	df_final = pd.DataFrame(rows_list)

	### Export summary dataframe to CSV
	fn = 'HedonicOutputs_Summary.csv'
	final_csv_uri = os.path.join(paths.outputs_dir, fn)
	df_final.to_csv(final_csv_uri, index=False)

	return None


def calculate_overvaluation(
	assume_nonsfha_discount=False, assume_xs_discount=False):

	"""Calculate potential overvaluation.

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

	### Read county groups CSV to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Get model outputs
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	coeff_100yr_group_a = float(model_a['100yr_coeff'].dropna())
	coeff_100yr_group_b = float(model_b['100yr_coeff'].dropna())
	coeff_100yr_group_c = float(model_c['100yr_coeff'].dropna())
	coeff_100yr_group_d = float(model_d['100yr_coeff'].dropna())

	### Get list of available counties
	fn_list = os.listdir(paths.ztrax_cleaned_csv_dir)
	csv_uri_list = []
	for fn in fn_list:
		csv_uri = os.path.join(paths.ztrax_cleaned_csv_dir, fn)
		csv_uri_list.append(csv_uri)

	### Summary rows list
	rows_list = []

	### Iterate through counties
	for fips_csv_uri in sorted(csv_uri_list):
		
		### Get county FIPS code
		fips = os.path.basename(fips_csv_uri).split('.')[0]

		print('\t\tCounty FIPS: %s' %fips)

		### Read county-level ZTRAX CSV to Pandas dataframe
		df = pd.read_csv(fips_csv_uri, low_memory=False)

		### Get county characteristics
		group = df_groups[df_groups['fips']==int(fips)]
		a = int(group['perc_worried_binary'])
		d = int(group['disclosure_binary'])

		### Get 100-year coefficient, based on county characteristics
		if a == 0 and d == 0:
			coeff = coeff_100yr_group_a

		if a == 0 and d == 1:
			coeff = coeff_100yr_group_b

		if a == 1 and d == 0:
			coeff = coeff_100yr_group_c

		if a == 1 and d == 1:
			coeff = coeff_100yr_group_d

		### Sort by property ID and year
		df = df.sort_values(['pid', 'date']).reset_index(drop=True)

		### Only include unique properties
		df = df.drop_duplicates(subset='pid', keep='last')

		### Add 100-yr coefficient as column in dataframe
		df['100-yr_coeff'] = np.where(df['flood_zone']=='100yr', coeff, np.nan)

		### Calculate price without empirical flood zone discount
		df['price_nonfz'] = np.where(df['flood_zone']=='100yr', 
									 df['fmv'] / (1 + coeff), 
									 df['fmv'])

		### If 'assume_nonsfha_discount' is True...
		if assume_nonsfha_discount == True:
			### Calculate price without empirical flood zone discount
			df['price_nonfz'] = df['fmv'] / (1 + coeff)

		### If 'assume_xs_discount' is True...
		if assume_xs_discount == True:
			coeff_100yr = utils.get_xs_coeff('100', a, d)
			coeff_500yr = utils.get_xs_coeff('500', a, d)
			coeff_outside = utils.get_xs_coeff('outside', a, d)

			### Calculate price without empirical flood zone discount
			df['price_nonfz'] = np.where(df['flood_zone']=='100yr', 
									 df['fmv'] / (1 + coeff_100yr), 
									 df['fmv'])

			df['price_nonfz'] = np.where(df['flood_zone']=='500yr', 
									 df['fmv'] / (1 + coeff_500yr), 
									 df['price_nonfz'])

			df['price_nonfz'] = np.where(df['flood_zone']=='outside', 
									 df['fmv'] / (1 + coeff_outside), 
									 df['price_nonfz'])

		### Calculate NPV of damages for multiple discount rates
		discount_rates = [1, 3, 5, 7]

		### Iterate through EAL columns
		for eal_col in reference.eal_cols:
			eal_col_label = eal_col.replace(r'%s_', '')
			eal_col_2020 = eal_col %'2020'
			eal_col_2050 = eal_col %'2050'

			### Iterate through discount rates
			for dr in discount_rates:	

				### Calculate NPV of losses over 30-year time horizon
				npv_col = 'npv_dr%d_%s' %(dr, eal_col_label)
				df[npv_col] = utils.npv(df, eal_col_2020, eal_col_2050, dr)

				### Calculate efficient price, based on NPV of damages subtracted
				### from price without empirical flood zone discount ('price_nonfz')
				effprice_col = 'effprice_dr%d_%s' %(dr, eal_col_label)
				df[effprice_col] = df['price_nonfz'] - df[npv_col]

				df[effprice_col] = np.where(df[effprice_col]<0, 0, df[effprice_col])			

				### Calculate efficient discount, based on the percent difference
				### between the efficient price and the price without empirical 
				### flood zone discount
				effdiscount_col = 'effdiscount_dr%d_%s' %(dr, eal_col_label)
				df[effdiscount_col] = ((df[effprice_col] - df['price_nonfz']) / 
										df['price_nonfz'])

				### Calculate percentage by which properties are overvalued, based
				### on the percent difference between the empirical price and the 
				### efficient price
				perc_overvaluation_col = 'perc_overvalue_dr%d_%s' %(dr, eal_col_label)
				df[perc_overvaluation_col] = ((df[effprice_col] - df['fmv']) /
											  df['fmv']) * -1

				### Calculate the total monetary value of potential overvaluation,
				### based on the difference between the empirical price and the 
				### efficient price
				monetary_overvaluation_col = 'dollars_overvalue_dr%d_%s' %(dr, eal_col_label)
				df[monetary_overvaluation_col] = df['fmv'] - df[effprice_col]


		### Export county-level dataframe to CSV
		fn = '%s.csv' %fips
		county_csv_uri = os.path.join(paths.outputs_dir, 'County_CSVs', fn)

		if assume_nonsfha_discount == True:
			county_csv_uri = os.path.join(
				paths.outputs_dir, 'County_CSVs_AssumeNonSFHAdiscount', fn)
		
		if assume_xs_discount == True:
			county_csv_uri = os.path.join(
				paths.outputs_dir, 'County_CSVs_AssumeXSdiscount', fn)

		df.to_csv(county_csv_uri, index=False)

	return None


def summarize_countyoutputs(
	assume_nonsfha_discount=False, assume_xs_discount=False):

	"""Summarize county-level outputs.

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

	### Read county groups CSV to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Get model outputs
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	coeff_100yr_group_a = float(model_a['100yr_coeff'].dropna())
	coeff_100yr_group_b = float(model_b['100yr_coeff'].dropna())
	coeff_100yr_group_c = float(model_c['100yr_coeff'].dropna())
	coeff_100yr_group_d = float(model_d['100yr_coeff'].dropna())

	### Initialize path to directory with county output CSV files
	county_csv_dir = os.path.join(paths.outputs_dir, 'County_CSVs')

	if assume_nonsfha_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeNonSFHAdiscount')

	if assume_xs_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeXSdiscount')

	### Initialize rows list
	rows_list = []

	### Iterate through CSV filenames in output directory
	for csv_fn in sorted(os.listdir(county_csv_dir)):
		
		### Get county FIPS code
		fips = os.path.basename(csv_fn).split('.')[0]
		print('\t\tCounty FIPS: %s' %fips)

		### Initialize full path to CSV file
		csv_uri = os.path.join(county_csv_dir, csv_fn)

		### Read to CSV file to Pandas dataframe
		df = pd.read_csv(csv_uri, low_memory=False)

		### Get county characteristics
		group = df_groups[df_groups['fips']==int(fips)]
		a = int(group['perc_worried_binary'])
		d = int(group['disclosure_binary'])

		### Get 100-year coefficient, based on county characteristics
		if a == 0 and d == 0:
			coeff_100yr = coeff_100yr_group_a

		if a == 0 and d == 1:
			coeff_100yr = coeff_100yr_group_b

		if a == 1 and d == 0:
			coeff_100yr = coeff_100yr_group_c

		if a == 1 and d == 1:
			coeff_100yr = coeff_100yr_group_d

		### Initialize list of discount rates
		discount_rates = [1, 3, 5, 7]

		### Iterate through EAL columns
		for eal_col in reference.eal_cols:
			eal_col_label = eal_col.replace(r'%s_', '')

			### Iterate through discount rates
			for dr in discount_rates:	

				### Get column names
				npv_col = 'npv_dr%d_%s' %(dr, eal_col_label)
				effprice_col = 'effprice_dr%d_%s' %(dr, eal_col_label)
				effdiscount_col = 'effdiscount_dr%d_%s' %(dr, eal_col_label)
				perc_overvaluation_col = 'perc_overvalue_dr%d_%s' %(dr, eal_col_label)
				monetary_overvaluation_col = 'dollars_overvalue_dr%d_%s' %(dr, eal_col_label)

				### Replace NaN values with zeros
				df[npv_col] = df[npv_col].fillna(0)
				df[effprice_col] = df[effprice_col].fillna(0)
				df[effdiscount_col] = df[effdiscount_col].fillna(0)
				df[perc_overvaluation_col] = df[perc_overvaluation_col].fillna(0)
				df[monetary_overvaluation_col] = df[monetary_overvaluation_col].fillna(0)

				### Store output in row_dict
				row_dict = {
					
					######################## Identifiers #######################

					### County FIPS code
					'fips': fips,
					
					### 100-year coefficient
					'100-yr_coeff': coeff_100yr,

					### AAL method
					'eal_method': eal_col_label,
					
					### Applied discount rate
					'discount_rate': dr,

					################### Number of properties ###################
					
					### All properties
					'n_prop': len(df['pid'].unique()),
					
					### SFHA properties
					'n_prop_sfha': len(df['pid'][df['flood_zone']=='100yr']),
					
					### Non-SFHA properties
					'n_prop_nonsfha': len(df['pid'][df['flood_zone']!='100yr']),
					
					### Properties with NPV > 0
					'n_prop_npv>0': len(df['pid'][df[npv_col]>0]),
					
					### SFHA properties with NPV > 0
					'n_prop_sfha_npv>0': len(df['pid'][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]),
					
					### Non-SFHA properties with NPV > 0
					'n_prop_nonsfha_npv>0': len(df['pid'][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]),

					############## Fair market value of properties #############
					
					### All properties
					'fmv_mean': df['fmv'].mean(),
					'fmv_median': df['fmv'].median(),
					'fmv_sum': df['fmv'].sum(),
								
					### All properties
					'fmv_nonfz_mean': df['price_nonfz'].mean(),
					'fmv_nonfz_median': df['price_nonfz'].median(),
					
					############### Net present value of damages ###############
					
					### All properties
					'npv_mean': df[npv_col].mean(),
					'npv_median': df[npv_col].median(),
					'npv_sum': df[npv_col].sum(),
					
					### Properties with NPV > 0
					'npv_npv>0_mean': (df[npv_col][df[npv_col]>0]).mean(),
					'npv_npv>0_median': (df[npv_col][df[npv_col]>0]).median(),
					'npv_npv>0_sum': (df[npv_col][df[npv_col]>0]).sum(),

					### SFHA properties with NPV > 0
					'npv_sfha_npv>0_mean': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).mean(),
					'npv_sfha_npv>0_median': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).median(),
					'npv_sfha_npv>0_sum': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).sum(),

					### Non-SFHA properties with NPV > 0
					'npv_nonsfha_npv>0_mean': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).mean(),
					'npv_nonsfha_npv>0_median': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).median(),
					'npv_nonsfha_npv>0_sum': (df[npv_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).sum(),

					##################### Efficient price ######################
					
					### All properties
					'effprice_mean': df[effprice_col].mean(),
					'effprice_median': df[effprice_col].median(),
					
					################# Efficient price discount #################
					
					### All properties
					'effdiscount_mean': df[effdiscount_col].mean(),
					'effdiscount_median': df[effdiscount_col].median(),
					
					### Properties with NPV > 0
					'effdiscount_npv>0_mean': (df[effdiscount_col][
						df[npv_col]>0]).mean(),
					'effdiscount_npv>0_median': (df[effdiscount_col][
						df[npv_col]>0]).median(),

					### SFHA properties with NPV > 0
					'effdiscount_sfha_npv>0_mean': (df[effdiscount_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).mean(),
					'effdiscount_sfha_npv>0_median': (df[effdiscount_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).median(),

					### Non-SFHA properties with NPV > 0
					'effdiscount_nonsfha_npv>0_mean': (df[effdiscount_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).mean(),
					'effdiscount_nonsfha_npv>0_median': (df[effdiscount_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).median(),					

					#################### Percent overvalued ####################

					### All properties
					'percovervalued_mean': df[perc_overvaluation_col].mean(),
					'percovervalued_median': df[perc_overvaluation_col].median(),
					
					### Properties with NPV > 0
					'percovervalued_npv>0_mean': (df[perc_overvaluation_col][
						df[npv_col]>0]).mean(),
					'percovervalued_npv>0_median': (df[perc_overvaluation_col][
						df[npv_col]>0]).median(),

					### SFHA properties with NPV > 0
					'percovervalued_sfha_npv>0_mean': 
						(df[perc_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).mean(),
					'percovervalued_sfha_npv>0_median': 
						(df[perc_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).median(),

					### Non-SFHA properties with NPV > 0
					'percovervalued_nonsfha_npv>0_mean': 
						(df[perc_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).mean(),
					'percovervalued_nonsfha_npv>0_median': 
						(df[perc_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).median(),	

					################### Dollar overvaluation ###################
					
					### All properties
					'monetary_overvaluation_mean': 
						df[monetary_overvaluation_col].mean(),
					'monetary_overvaluation_median': 
						df[monetary_overvaluation_col].median(),
					'monetary_overvaluation_sum': 
						df[monetary_overvaluation_col].sum(),

					### Properties with NPV > 0
					'monetary_overvaluation_npv>0_mean': 
						(df[monetary_overvaluation_col][
						df[npv_col]>0]).mean(),
					'monetary_overvaluation_npv>0_median': 
						(df[monetary_overvaluation_col][
						df[npv_col]>0]).median(),
					'monetary_overvaluation_npv>0_sum': 
						(df[monetary_overvaluation_col][
						df[npv_col]>0]).sum(),	

					### SFHA properties with NPV > 0
					'monetary_overvaluation_sfha_npv>0_mean': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).mean(),
					'monetary_overvaluation_sfha_npv>0_median': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).median(),
					'monetary_overvaluation_sfha_npv>0_sum': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']=='100yr')]).sum(),

					### Non-SFHA properties with NPV > 0
					'monetary_overvaluation_nonsfha_npv>0_mean': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).mean(),
					'monetary_overvaluation_nonsfha_npv>0_median': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).median(),	
					'monetary_overvaluation_nonsfha_npv>0_sum': 
						(df[monetary_overvaluation_col][
						(df[npv_col]>0) & (df['flood_zone']!='100yr')]).sum(),	
					}

				### Append row_dict to rows_list
				rows_list.append(row_dict)

	### Export summary dataframe to CSV
	df_final = pd.DataFrame(rows_list)
	final_csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')

	if assume_nonsfha_discount == True:
		final_csv_uri = os.path.join(
			paths.outputs_dir, 'Outputs_Summary_AssumeNonSFHAdiscount.csv')

	if assume_xs_discount == True:
		final_csv_uri = os.path.join(
			paths.outputs_dir, 'Outputs_Summary_AssumeXSdiscount.csv')

	df_final.to_csv(final_csv_uri, index=False)

	return None


def combine_atriskproperties(
	assume_nonsfha_discount=False, assume_xs_discount=False): 

	"""Combine all properties exposed to flood risk.

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

	### Initialize path to directory with county output CSV files
	county_csv_dir = os.path.join(paths.outputs_dir, 'County_CSVs')

	if assume_nonsfha_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeNonSFHAdiscount')

	if assume_xs_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeXSdiscount')

	### Initialize rows list
	df_list = []

	### Iterate through CSV filenames in output directory
	for csv_fn in sorted(os.listdir(county_csv_dir)):
		
		### Get county FIPS code
		fips = os.path.basename(csv_fn).split('.')[0]
		print('\t\tCounty FIPS: %s' %fips)

		### Initialize full path to CSV file
		csv_uri = os.path.join(county_csv_dir, csv_fn)

		### Read to CSV file to Pandas dataframe
		df_county = pd.read_csv(csv_uri, low_memory=False)

		### Subset dataframe columns
		df_county = df_county[[
			### Identifier columns
			'fips', 'geoid', 'pid', 'fmv', 

			### Flood zone columns
			'q3_floodzone', 'dfirm_floodzone', 'flood_zone',

			### Data columns with 1% discount rate
			'npv_dr1_fld_eal_base_noFR_mid_fs_m',
			'perc_overvalue_dr1_fld_eal_base_noFR_mid_fs_m',
			'dollars_overvalue_dr1_fld_eal_base_noFR_mid_fs_m',
			
			### Data columns with 3% discount rate
			'npv_dr3_fld_eal_base_noFR_mid_fs_m',
			'perc_overvalue_dr3_fld_eal_base_noFR_mid_fs_m',
			'dollars_overvalue_dr3_fld_eal_base_noFR_mid_fs_m',

			### Data columns with 5% discount rate
			'npv_dr5_fld_eal_base_noFR_mid_fs_m',
			'perc_overvalue_dr5_fld_eal_base_noFR_mid_fs_m',
			'dollars_overvalue_dr5_fld_eal_base_noFR_mid_fs_m',

			### Data columns with 7% discount rate
			'npv_dr7_fld_eal_base_noFR_mid_fs_m',
			'perc_overvalue_dr7_fld_eal_base_noFR_mid_fs_m',
			'dollars_overvalue_dr7_fld_eal_base_noFR_mid_fs_m',
			]]

		### Rename columns
		df_county = df_county.rename(columns={
			'npv_dr1_fld_eal_base_noFR_mid_fs_m': 'npv_dr1',
			'perc_overvalue_dr1_fld_eal_base_noFR_mid_fs_m': 'perc_overvalue_dr1',
			'dollars_overvalue_dr1_fld_eal_base_noFR_mid_fs_m': 'dollars_overvalue_dr1',

			'npv_dr3_fld_eal_base_noFR_mid_fs_m': 'npv_dr3',
			'perc_overvalue_dr3_fld_eal_base_noFR_mid_fs_m': 'perc_overvalue_dr3',
			'dollars_overvalue_dr3_fld_eal_base_noFR_mid_fs_m': 'dollars_overvalue_dr3',

			'npv_dr5_fld_eal_base_noFR_mid_fs_m': 'npv_dr5',
			'perc_overvalue_dr5_fld_eal_base_noFR_mid_fs_m': 'perc_overvalue_dr5',
			'dollars_overvalue_dr5_fld_eal_base_noFR_mid_fs_m': 'dollars_overvalue_dr5',

			'npv_dr7_fld_eal_base_noFR_mid_fs_m': 'npv_dr7',
			'perc_overvalue_dr7_fld_eal_base_noFR_mid_fs_m': 'perc_overvalue_dr7',
			'dollars_overvalue_dr7_fld_eal_base_noFR_mid_fs_m': 'dollars_overvalue_dr7',			
			})

		### Subset dataframe to only include properties exposed to flood risk
		df_county = df_county[df_county['npv_dr1']>0]

		### Append county dataframe to list of county dataframes
		df_list.append(df_county)

	### Concatenate dataframes in list of county dataframes
	df = pd.concat(df_list)

	### Export final dataframe to CSV file
	fn = 'Combined_AtRiskProperties.csv'

	if assume_nonsfha_discount == True:
		fn = 'Combined_AtRiskProperties_AssumeNonSFHAdiscount.csv'
	
	if assume_xs_discount == True:
		fn = 'Combined_AtRiskProperties_AssumeXSdiscount.csv'
	
	csv_uri = os.path.join(paths.outputs_dir, fn)
	df.to_csv(csv_uri, index=False)

	return None


def montecarlo_simulation():

	"""Run Monte Carlo simulation.

	Arguments:
		None

	Returns:
		None
	"""
	
	### Read county groups CSV to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Get model outputs
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	mean_a = float(model_a['100yr_coeff'].dropna())
	mean_b = float(model_b['100yr_coeff'].dropna())
	mean_c = float(model_c['100yr_coeff'].dropna())
	mean_d = float(model_d['100yr_coeff'].dropna())

	std_error_a = float(model_a['100yr_se'].dropna())
	std_error_b = float(model_b['100yr_se'].dropna())
	std_error_c = float(model_c['100yr_se'].dropna())
	std_error_d = float(model_d['100yr_se'].dropna())

	### Create normal distributions for 100-year coefficients
	coeff_100yr_group_a_distribution = stats.norm(mean_a, std_error_a)
	coeff_100yr_group_b_distribution = stats.norm(mean_b, std_error_b)
	coeff_100yr_group_c_distribution = stats.norm(mean_c, std_error_c)
	coeff_100yr_group_d_distribution = stats.norm(mean_d, std_error_d)

	### Initialize list of discount rates discount rates
	discount_rates = [1, 3, 5, 7]

	### Get list of available counties
	fn_list = os.listdir(paths.ztrax_cleaned_csv_dir)
	csv_uri_list = []
	for fn in fn_list:
		csv_uri = os.path.join(paths.ztrax_cleaned_csv_dir, fn)
		csv_uri_list.append(csv_uri)

	### Initialize number of iterations in Monte Carlo simulation
	n = 100

	### Seed random number generator
	np.random.seed(0)

	### Randomly sample normal distributions for 100-yr coefficients
	coeff_100yr_group_a = coeff_100yr_group_a_distribution.rvs(n)
	coeff_100yr_group_b = coeff_100yr_group_b_distribution.rvs(n)
	coeff_100yr_group_c = coeff_100yr_group_c_distribution.rvs(n)
	coeff_100yr_group_d = coeff_100yr_group_d_distribution.rvs(n)

	### Iterate through counties
	for fips_csv_uri in sorted(csv_uri_list)[-54:]:

		### Summary rows list
		rows_list = []

		### Get county FIPS code
		fips = os.path.basename(fips_csv_uri).split('.')[0]
		
		### Get county characteristics
		group = df_groups[df_groups['fips']==int(fips)]
		a = int(group['perc_worried_binary'])
		d = int(group['disclosure_binary'])

		### Get 100-year coefficient, based on county characteristics
		if a == 0 and d == 0:
			coeff_list = coeff_100yr_group_a

		if a == 0 and d == 1:
			coeff_list = coeff_100yr_group_b

		if a == 1 and d == 0:
			coeff_list = coeff_100yr_group_c

		if a == 1 and d == 1:
			coeff_list = coeff_100yr_group_d

		### Read county CSV fo Pandas dataframe
		df = pd.read_csv(fips_csv_uri, low_memory=False)

		### Sort by property ID and year
		df = df.sort_values(['pid', 'date']).reset_index(drop=True)

		### Only include unique properties
		df = df.drop_duplicates(subset='pid', keep='last')

		### Get flood zone column
		flood_zone = df['flood_zone']

		### Get fair market value column
		fmv = df['fmv']

		### Iterate through n simulations
		def monte_carlo(i):
			print('\t\tSimulation: %d of %d; County: %s' %(i+1, n, fips))

			### Initialize list to store results within simulation
			sim_list = []

			### Get coefficient from coefficient list
			coeff = coeff_list[i]

			### Iterate through EAL columns
			for eal_col in reference.eal_cols:
				eal_col_label = eal_col.replace(r'%s_', '')
				eal_col_2020 = eal_col %'2020'
				eal_col_2050 = eal_col %'2050'

				depthdamage_function = eal_col.split('_')[2]
				damage_scenario = eal_col.split('_')[4]
				hazard_scenario = eal_col.split('_')[7]

				### Iterate through discount rates
				for dr in discount_rates:	

					### Calculate price without empirical flood zone discount
					price_nonfz = np.where(
						flood_zone=='100yr', fmv / (1 + coeff), fmv)

					### Calculate NPV of losses over 30-year time horizon
					npv = utils.npv(df, eal_col_2020, eal_col_2050, dr)

					### Calculate efficient price
					eff_price = price_nonfz - npv
					eff_price = np.where(eff_price<0, 0, eff_price)			

					### Calculate the total monetary value of potential overvaluation
					overvaluation = fmv - eff_price

					### Remove negative values
					overvaluation = overvaluation[overvaluation>0]

					### Calculate total overvaluation within county
					overvaluation_sum = overvaluation.sum()

					### Store outputs in row dictionary:
					row_dict = {
						'fips': fips,
						'simulation': i,
						'coeff': coeff,
						'depthdamage_function': depthdamage_function,
						'damage_scenario': damage_scenario,
						'hazard_scenario': hazard_scenario,
						'discount_rate': dr,
						'overvaluation': overvaluation_sum,
						}

					### Append row_dict to rows_list
					sim_list.append(row_dict)

			return sim_list

		rows_list = Parallel(n_jobs=-1)(delayed(
			monte_carlo)(i) for i in range(n))

		### Flatten list of lists
		flat_list = [item for sublist in rows_list for item in sublist]

		### Convert list of row dictionaries to Pandas DataFrame
		df_final = pd.DataFrame(flat_list)

		### Export summary dataframe to CSV
		fn = '%s.csv' %fips
		final_csv_uri = os.path.join(paths.outputs_dir, 'MonteCarlo_Outputs', fn)
		df_final.to_csv(final_csv_uri, index=False)

	return None


def collect_montecarlo_outputs():

	"""Collect outputs from Monte Carlo simulation.

	Arguments:
		None

	Returns:
		None
	"""

	### Get list of available counties
	fn_list = os.listdir(os.path.join(paths.outputs_dir, 'MonteCarlo_Outputs'))
	csv_uri_list = []
	for fn in fn_list:
		csv_uri = os.path.join(paths.outputs_dir, 'MonteCarlo_Outputs', fn)
		csv_uri_list.append(csv_uri)

	### Create final dataframe to store outputs from first CSV file in list
	df_final = pd.read_csv(csv_uri_list[0])
	df_final = df_final[[
			'simulation', 'depthdamage_function', 'damage_scenario',
			'hazard_scenario', 'discount_rate']]

	### Sorted values in final dataframe
	df_final = df_final.sort_values(by=[
		'simulation', 'depthdamage_function', 'damage_scenario',
		'hazard_scenario', 'discount_rate']).reset_index(drop=True)

	### Create total overvaluation column
	df_final['overvaluation'] = 0

	### Iterate through county CSV files
	for csv_uri in sorted(csv_uri_list):

		### Get county FIPS code
		fips = os.path.basename(csv_uri).split('.')[0]
		print('\t\tCounty FIPS: %s' %fips)

		### Read CSV file Pandas dataframe
		df = pd.read_csv(csv_uri)

		### Sorted values in dataframe
		df = df.sort_values(by=[
			'simulation', 'depthdamage_function', 'damage_scenario',
	 		'hazard_scenario', 'discount_rate']).reset_index()

		### Add overvaluation for each county
		df_final['overvaluation'] += df['overvaluation']

	### Export final dataframe to CSV file
	csv_uri = os.path.join(paths.outputs_dir, 'MonteCarlo_Summary.csv')
	df_final.to_csv(csv_uri, index=False)

	return None


def get_censustract_outputs(
	assume_nonsfha_discount=False, assume_xs_discount=False):

	"""Get census tract-level outputs.

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

	### Initialize path to directory with county output CSV files
	county_csv_dir = os.path.join(paths.outputs_dir, 'County_CSVs')

	if assume_nonsfha_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeNonSFHAdiscount')

	if assume_xs_discount == True:
		county_csv_dir = os.path.join(
			paths.outputs_dir, 'County_CSVs_AssumeXSdiscount')

	### Initialize rows list
	rows_list = []

	### Iterate through CSV filenames in output directory
	for csv_fn in sorted(os.listdir(county_csv_dir)):
		
		### Get county FIPS code
		fips = os.path.basename(csv_fn).split('.')[0]
		print('\t\tCounty FIPS: %s' %fips)

		### Initialize full path to CSV file
		csv_uri = os.path.join(county_csv_dir, csv_fn)

		### Read to CSV file to Pandas dataframe
		df = pd.read_csv(csv_uri, low_memory=False)

		### Initialize list of discount rates
		discount_rates = [1, 3, 5, 7]

		### Remove nan tracts from df
		df = df[df['tract']!='000nan']

		### Iterate through census tracts
		for t in df['tract'].unique(): 

			### Subset dataframes to tract t
			df_tract = df[df['tract']==t]

			### Get summary data
			n_properties = len(df_tract)
			total_fmv = df_tract['fmv'].sum()

			### Store summary data in a dictionary
			row_dict = {
				'county': int(fips),
				'tract': int(float(t)),
				'n_properties': n_properties,
				'total_fmv': total_fmv,
				}

			### Iterate through applied discount rates
			for discount_rate in [1, 3, 5, 7]:

				### Get names of overvaluation columns in dataframe
				eal_col_label = 'fld_eal_base_noFR_mid_fs_m'
				monetary_overvaluation_col = 'dollars_overvalue_dr%d_%s' %(
					discount_rate, eal_col_label)
				perc_overvaluation_col = 'perc_overvalue_dr%d_%s' %(
					discount_rate, eal_col_label)

				### Get total overvaluation for census tract
				total_overvaluation = df_tract[monetary_overvaluation_col].sum()
				
				### Insert overvaluation data in row_dict
				overvaluation_key = 'total_ov_%d' %discount_rate
				row_dict[overvaluation_key] = total_overvaluation

				### Iterate through overvaluation bins 
				for bin_lower in np.arange(0, 1, 0.05):
					
					### Subset dataframe to only include properties that are 
					### overvalued by bin_lower to (bin_lower + 0.05)
					bin_upper = bin_lower + 0.05

					df_tract_bin = df_tract[
						(df_tract[perc_overvaluation_col] >  bin_lower) &
						(df_tract[perc_overvaluation_col] <= bin_upper)]
					
					bin_key = ('n_properties_ov%dto%d_%d' 
						%(bin_lower*100, bin_upper*100, discount_rate))
					row_dict[bin_key] = len(df_tract_bin)

			### Append row_dict to rows_list
			rows_list.append(row_dict)

	### Convert rows list to Pandas dataframe
	df_final = pd.DataFrame(rows_list)

	### Export summary dataframe to CSV
	csv_fn = 'CensusTract_Outputs.csv'
	
	if assume_nonsfha_discount == True:
		csv_fn = 'CensusTract_Outputs_AssumeNonSFHAdiscount.csv'

	if assume_xs_discount == True:
		csv_fn = 'CensusTract_Outputs_AssumeXSdiscount.csv'

	final_csv_uri = os.path.join(paths.outputs_dir, csv_fn)
	df_final.to_csv(final_csv_uri, index=False)

	return None


def fiscalvulnerability_analysis():

	"""Run fiscal vulnerability analysis.

	Arguments:
		None
	
	Returns:
		None
	"""

	### Read outputs summary CSV to Pandas DataFrame
	df1 = pd.read_csv(os.path.join(paths.outputs_dir, 'Outputs_Summary.csv'))
	
	### Read municipal finance CSV to Pandas DataFrame
	df2 = pd.read_csv(paths.munifinance_csv_uri)

	######################## Prep municipal finance data #######################
	
	### Subset columns in dataframe
	df2 = df2[['FIPSid', 'Year4', 'County', 'FIPS_Combined', 
			   'Total_Revenue', 'Property_Tax']]

	### Remove rows with NaN data in FIPS_Combined column
	df2 = df2[~df2['FIPS_Combined'].isna()]

	### Change FIPS_Combined data type from float to int
	df2['FIPS_Combined'] = df2['FIPS_Combined'].astype(int)

	### Sort values based on year
	df2 = df2.sort_values('Year4')

	### Drop FIPSid (i.e. municipalities) duplicates, except for last entry
	### This selects for only the most recent year of data
	df2 = df2.drop_duplicates('FIPSid', keep='last')

	### Groupby county and sum data
	df2 = df2.groupby('FIPS_Combined', as_index=False).sum()	

	### Subset columns in dataframe
	df2 = df2[['FIPS_Combined', 'Total_Revenue', 'Property_Tax']]

	############################ Merge df1 and df2 #############################
	
	df = df1.merge(df2, how='inner', left_on='fips', right_on='FIPS_Combined')

	############################ Clean up dataframe ############################
	
	### Drop FIPS_Combined column
	df = df.drop('FIPS_Combined', axis=1)
	
	### Rename columns
	df = df.rename(columns={
		'Total_Revenue': 'total_revenue',
		'Property_Tax': 'property_tax'
		})

	### Calculate percent revenue from property taxes
	df['percrev_proptax'] = df['property_tax'] / df['total_revenue']

	### Subset data to 3% discount rate
	df = df[df['eal_method']=='fld_eal_base_noFR_mid_fs_m']
	df = df[df['discount_rate']==3]

	### Read county FIPS codes CSV to dictionary
	fips_codes_csv_uri = os.path.join(
		paths.data_dir, 'FIPS_Codes/county_fips_master.csv')
	df_fipscodes = pd.read_csv(fips_codes_csv_uri)

	### Merge FIPS codes dataframe to primary dataframe
	df = df.merge(df_fipscodes, how='left', on='fips')

	### Get columns of interest
	df['sx'] = df['percrev_proptax']
	df['sy'] = df['monetary_overvaluation_sum'] / df['fmv_sum']

	print(df.columns)

	### Subset columns in dataframe
	df = df[['fips', 'county_name', 'state_abbr', 'sx', 'sy', 'total_revenue', 
			 'percrev_proptax', 'property_tax', 'monetary_overvaluation_sum', 'fmv_sum',
			 'percovervalued_npv>0_median']]

	### Subset dataframe to at-risk counties 
	df_atrisk = df[(df['sx']>=np.nanpercentile(df['sx'], 80)) & 
			 	   (df['sy']>=np.nanpercentile(df['sy'], 80))]
	
	### Print at-risk counties dataframe
	print(df_atrisk[['county_name', 'state_abbr', 'sx', 'sy']].sort_values(
		['state_abbr', 'county_name']))

	### Export df_atrisk to CSV file
	df_atrisk['County'] = df_atrisk['county_name'] + ', ' + df_atrisk['state_abbr']
	fn = 'Fiscal_Vulnerability_AtRisk.csv'
	output_csv_uri = os.path.join(paths.outputs_dir, fn)
	df_atrisk.to_csv(output_csv_uri)

	### Export dataframe to CSV
	fn = 'Fiscal_Vulnerability.csv'
	output_csv_uri = os.path.join(paths.outputs_dir, fn)
	df.to_csv(output_csv_uri, index=False)

	return None


def collect_keyoutputs(discount_rate=3, hazard_scenario='m'):

	"""Get census tract-level outputs.

	Arguments:
		None
	
	Option arguments:
		discount_rate (int): Applied discount rate as an integer 
		hazard_scenario (string): Low ('l'), medium ('m'), or high 
			('h') hazard scenario

	Returns:
		None
	"""

	### Read outputs summary to Pandas dataframe
	outputs_csv_uri = os.path.join(paths.outputs_dir, 'Outputs_Summary.csv')
	df = pd.read_csv(outputs_csv_uri)

	### Subset data to 3% discount rate
	df = df[df['discount_rate']==discount_rate]

	### Subset data to EAL method 
	eal_method = 'fld_eal_base_noFR_mid_fs_%s' %hazard_scenario
	df = df[df['eal_method']==eal_method]

	############## GET ESTIMATES FOR EMPIRICAL FLOOD ZONE DISCOUNT #############

	### Read county groups CSV to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Get model outputs
	model_a, model_b, model_c, model_d = utils.get_modeloutputs()

	### Initialize 100-yr coefficients and standard errors
	coeff_100yr_group_a = float(model_a['100yr_coeff'].dropna())
	coeff_100yr_group_b = float(model_b['100yr_coeff'].dropna())
	coeff_100yr_group_c = float(model_c['100yr_coeff'].dropna())
	coeff_100yr_group_d = float(model_d['100yr_coeff'].dropna())

	### Create 'coeff' column in data frame 
	df['coeff'] = np.nan

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

	df['coeff'] = np.where(
		df['fips'].isin(group_a_fips), coeff_100yr_group_a, df['coeff'])
	df['coeff'] = np.where(
		df['fips'].isin(group_b_fips), coeff_100yr_group_b, df['coeff'])
	df['coeff'] = np.where(
		df['fips'].isin(group_c_fips), coeff_100yr_group_c, df['coeff'])
	df['coeff'] = np.where(
		df['fips'].isin(group_d_fips), coeff_100yr_group_d, df['coeff'])


	######################### GET ADDITIONAL VARIABLES #########################

	col_name = 'overvaluation_percentoftotal'
	df[col_name] = (df['monetary_overvaluation_sum'] / df['fmv_sum']) * 100


	######################## ADD MUNICIPAL FINANCE DATA ########################

	### Initialize path to municipal finance CSV file
	munifinance_fn = 'Fiscal_Vulnerability.csv'
	munifinance_csv_uri = os.path.join(paths.outputs_dir, munifinance_fn)

	### Read municipal finance CSV to Pandas DataFrame
	df_muni = pd.read_csv(munifinance_csv_uri)

	### Subset municipal finance dataframe columns
	df_muni = df_muni[['fips', 'county_name', 'state_abbr', 'sx']]

	### Merge df and df_muni
	df = df.merge(df_muni, on='fips', how='left')


	########################## DROP UNNECESSARY COLUMNS ########################

	cols_ofinterest = [
		'fips', 
		'county_name',
		'state_abbr', 
		'coeff', 
		'percovervalued_npv>0_median',
		'overvaluation_percentoftotal',
		'monetary_overvaluation_sum',
		'sx',
		'fmv_median',
		'npv_sum'
		]

	df = df[cols_ofinterest]

	############################## RENAME COLUMNS ##############################

	df = df.rename(columns={
		'fips': 'fips',
		'county_name': 'name',
		'state_abbr': 'state',
		'coeff': 'empirical_fz_discount', 
		'percovervalued_npv>0_median': 'overvaluation_median',
		'overvaluation_percentoftotal': 'overvaluation_percfmv',
		'monetary_overvaluation_sum': 'overvaluation_dollars',
		'sx': 'perc_revenue_propertytax',
		'fmv_median': 'fmv_median',
		'npv_sum': 'npv_sum'
		})

	######################### WRITE COLUMN DESCRIPTIONS ########################

	col_descriptions = [
		['fips', 'County FIPS code'],
		
		['name', 'County name'],
		
		['state', 'State abbreviation'],
		
		['empirical_fz_discount', 
			'Estimated flood zone property price discount', 'Fig. 1A'],
		
		['overvaluation_median', 
			"Median overvaluation of properties exposed to flood risk, as a" + 
				"percentage of properties' current fair market value", 
			'Fig. 1B'],
		
		['overvaluation_percfmv', 
			'Property overvaluation as a percentage of the total fair market' + 
				'value of all properties', 
			'Fig. 1C'],
		
		['overvaluation_dollars', 
			'Total overvaluation in dollars', 'Fig. 1D'],
		
		['perc_revenue_propertytax', 
			'Percentage of municipal revenue derived from property taxes', 
			'Fig. 4 (horizontal axis; orange hues)'],
		
		['fmv_median', 'Median fair market property values', 'Fig. S2'],
		
		['npv_sum', 
			'Total net present value of average annual flood losses', 
			'Fig. S3'],
		]

	df_cols = pd.DataFrame(
		col_descriptions, columns=[
		'Column Name', 'Column Description', 'Corresponding Figure'])	

	################################## EXPORT ##################################

	### Export dataframe to CSV
	fn = 'CountyOutputs_toShare.csv'
	output_csv_uri = os.path.join(paths.outputs_dir, fn)
	df.to_csv(output_csv_uri, index=False)

	### Export column descriptions to CSV
	fn = 'CountyOutputs_toShare_ColumnDescriptions.xlsx'
	output_csv_uri = os.path.join(paths.outputs_dir, fn)
	df_cols.to_excel(output_csv_uri, index=False)

	return None
