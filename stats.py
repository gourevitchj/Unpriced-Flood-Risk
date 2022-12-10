### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: stats.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os
import pickle
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm

from linearmodels.panel import PanelOLS

import paths, reference

def execute(model_type):

	"""Execute series of functions for fitting panel model.

	Arguments:
		model_type (string): Type of model ('100yr' or '500yr')

	Returns:
		None
	"""

	### Read county groupings CSV file to Pandas dataframe
	df_groups = pd.read_csv(paths.countygroups_csv_uri)

	### Create list to store dataframe
	df_list = []

	### Iterate through counties
	print('\t\tReading in county CSV files')
	for csv_fn in sorted(os.listdir(paths.ztrax_cleaned_csv_dir)):
		
		### Print CSV filename
		print('\t\t\t%s' %csv_fn)

		### Get county CSV URI
		csv_uri = os.path.join(paths.ztrax_cleaned_csv_dir, csv_fn)

		### Read county-level ZTRAX CSV to Pandas dataframe
		df_county = pd.read_csv(csv_uri, low_memory=False)

		### Subset df_county to columns of interest
		cols = ['fips', 'pid', 'year', 'date', 'flood_zone', 'price_real']
		df_county = df_county[cols]

		### Append df_county to df_list  
		df_list.append(df_county)  

	### Concatenate df_county to df
	print('\t\tConcatenating dataframes')
	df = pd.concat(df_list)
	
	### Delete df_list from memory
	del df_list

	### Filter data
	print('\t\tFiltering data')
	df = filter_data(df, model_type)

	### Set coastal conditional
	c = 'x'

	### Generate itertools object to iterate through group subsets
	itertools_obj = itertools.product(['x', '0', '1'], ['x', '0' , '1'])

	### Iterate through itertools object
	for a, d in itertools_obj:

		### Set attitude conditional
		attitude_col = 'perc_personal_binary'
		if a=='x':
			a_conditional = (df_groups[attitude_col]>=0)
		if a=='0':
			a_conditional = (df_groups[attitude_col]==0)
		if a=='1':
			a_conditional = (df_groups[attitude_col]==1)
		
		### Set disclosure condition
		disclosure_col = 'disclosure_binary'
		if d=='x':
			d_conditional = (df_groups[disclosure_col]>=0)
		if d=='0':
			d_conditional = (df_groups[disclosure_col]==0)
		if d=='1':
			d_conditional = (df_groups[disclosure_col]==1)

		### Get FIPS codes for subset of counties
		fips_subset = list(df_groups['fips'][a_conditional & d_conditional])

		### Print grouping
		print('\t\tAttitude: %s; Coastal: %s; Disclosure: %s' %(a, c, d))
		print('\t\tN Counties in Sample: %d\n' %len(fips_subset))

		### Subset full dataframe to only counties within group
		print('\t\t\tSubsetting dataframe to counties in group')
		df_subset = df[df['fips'].isin(fips_subset)]

		### Run model
		print('\t\t\tFitting panel regression model')
		model = fit_model(df_subset, model_type)

		### Export model outputs
		print('\t\t\tExporting model outputs to pickle')
		export_modeloutputs(model, a, c, d, model_type) 
				
	return None


def filter_data(df, model_type):

	"""Filter data.

	Arguments:
		df (Pandas DataFrame): data
		model_type (string): Type of model ('100yr' or '500yr')

	Returns:
		None
	"""

	def execute_filter(df, model_type):

		"""Execute a series of data filtering functions.

		Arguments:
			df (Pandas DataFrame): data
			model_type (string): Type of model ('100yr' or '500yr')

		Returns:
			df (Pandas DataFrame): data
		"""

		### 1) Remove properties that have never been transacted
		df = remove_nevertransacted_properties(df)
		
		### 2) Sort dataframe
		df = sort_dataframe(df)

		### 3) Remove sales prior to 1996
		df = remove_earlysales(df)

		### 4) Sales where the floodplain status is unknown are dropped 
		df = remove_unmapped(df)
		
		### 5) Subset transactions by price
		df = subset_transactions_byprice(df)

		### 6) Must be sold more than once 
		df = remove_singletransaction_properties(df)
		
		### 7) Must be outside flood zone in old map
		df = subset_outsidefz_initialsale(df, model_type)
		
		### 8) Remove properties that exhibit more than 50% annual growth or 
		### decline in sale price between observed transactions 
		df = remove_outliers(df)

		return df


	def remove_nevertransacted_properties(df):
		
		"""Remove transactions that do not have transaction prices.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

		### Remove properties that have never been transacted
		df = df[~df['price_real'].isna()]

		return df


	def sort_dataframe(df):

		"""Sort data by property ID and date of transaction.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

	    ### Sort dataframe by 'pid' and 'date'
	    df = df.sort_values(['pid', 'date']).reset_index(drop=True)

	    return df


	def remove_earlysales(df):

		"""Remove transactions that occurred before 1996.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

		### Remove transactions that occurred prior to 1996
		### 1996 is what Hino & Burke (2021) use
		df = df[df['year']>=1996] 

		return df


	def remove_unmapped(df):

		"""Remove transactions where the floodzone is unmapped.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

	    ### Remove transactions where the flood zone is unknown
	    df = df[df['flood_zone'] != 'unmapped']

	    return df


	def subset_transactions_byprice(df):

		"""Remove transaction where the floodzone is unmapped.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

		### Subset data to transactions with prices greater than $10,000
		df = df[df['price_real']>=10000]

		return df


	def remove_singletransaction_properties(df):

		"""Remove properties with only one transaction.

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

	    ### Remove properties with only one transactions
	    df_duplicated = df[df.duplicated(subset='pid', keep=False)]
	    df = df[df['pid'].isin(df_duplicated['pid'].unique())]

	    return df


	def subset_outsidefz_initialsale(df, model_type):

		"""Only include properties that were outside of the 100-yr 
			floodplain at the time of their first sale in the dataset.

		Arguments:
			df (Pandas DataFrame): data
			model_type (string): Type of model ('100yr' or '500yr')

		Returns:
			df (Pandas DataFrame): data
		"""
		
		### Subset dataframe to only include first transactions
		df_first = df.drop_duplicates(subset='pid', keep='first')
	
		### If 100-yr model...
		if model_type=='100yr':	
			### Get transactions that occurred when property was outside SFHA
			df_first_outside = df_first[
				(df_first['flood_zone']=='outside') | 
				(df_first['flood_zone']=='500yr')
				]

		### If 500-yr model...
		if model_type=='500yr':	
			### Get transactions that occurred when property was outside flood zone
			df_first_outside = df_first[
				(df_first['flood_zone']=='outside')
				]

		### Get property IDs where first transaction was outside flood zone
		pid_subset = df_first_outside['pid'].unique()

		### Subset dataframe to only include these properties
		df = df[df['pid'].isin(pid_subset)]

		return df


	def remove_outliers(df):
		
		"""Remove properties that exhibit more than 50% annual growth or 
			decline in sale price between observed transactions

		Arguments:
			df (Pandas DataFrame): data

		Returns:
			df (Pandas DataFrame): data
		"""

		### Create empty to store property IDs to drop
		pid_droplist = []

		### Iterate through unique property IDs
		for pid in df['pid'].unique():
			df_pid = df[df['pid']==pid].reset_index(drop=True)

			### Get percent change in price between transactions
			df_pid['price_percchange'] = (
				(df_pid['price_real'] - df_pid['price_real'].shift(1)) / 
										df_pid['price_real'].shift(1))

			### Get number of years between transactions
			df_pid['years_betweensale'] = (df_pid['year'] - 
										   df_pid['year'].shift(1))

			### Get annualized percent change in price between transactions
			df_pid['price_percchange_annual'] = (df_pid['price_percchange'] / 
											  	 df_pid['years_betweensale'])

			### If annualized percent change in price between transactions is
			### less than -50% or greater than 50%, add pid to drop list
			if ((df_pid['price_percchange_annual'].min() < -0.5) or 
				(df_pid['price_percchange_annual'].max() >  0.5)):

				pid_droplist.append(pid)

		### Remove properties in drop list from dataframe
		df = df[~df['pid'].isin(pid_droplist)]

		return df

	### Execute series of data filtering functions
	df = execute_filter(df, model_type)

	return df


def fit_model(df_subset, model_type):

	"""Fit panel model.

	Arguments:
		df_subset (Pandas DataFrame): data
		model_type (string): Type of model ('100yr' or '500yr')

	Returns:
		model (object): fitted model
	"""
	
	### Create county-by-year column
	df_subset['county-year'] = (df_subset['fips'].astype(str) + 
						 		df_subset['year'].astype(int).astype(str))
	df_subset['county-year'] = df_subset['county-year'].astype(int)

	### Set dataframe index for entity and time fixed effects
	df_subset = df_subset.set_index(['pid', 'county-year'])

	### Create flood zone dummy variables 
	df_subset['100yr_dummy'] = np.where(df_subset['flood_zone']=='100yr', 1, 0)
	df_subset['500yr_dummy'] = np.where(df_subset['flood_zone']=='500yr', 1, 0)

	### Set name of independent variable columns
	if model_type == '100yr':
		X_cols = ['100yr_dummy']
	
	if model_type == '500yr':	
		X_cols = ['500yr_dummy']

	### Set name of dependent variable column
	y_col = 'price_real'

	### Subset dataframe to dependent and independent variable columns
	df_subset = df_subset[[y_col]+X_cols]
	
	### Remove NaN values
	df_subset = df_subset.dropna()

	### Set dependent variable
	y = df_subset[y_col]

	### Log transform price variable
	y = np.log(y)

	### Set independent variables
	X = df_subset[X_cols]

	### Add constant to independent variables
	X = sm.add_constant(X)

	### Fit model
	model = PanelOLS(y, X, entity_effects=True, time_effects=True)
	model = model.fit(low_memory=True)

	### Print model summary
	print(model)

	return model


def export_modeloutputs(model, a, c, d, model_type):

	"""Export model outputs to pickle.

	Arguments:
		model (object): fitted model
		a (string): climate attitudes conditional ('x', '0', '1')
		c (string): coastal conditional ('x', '0', '1')
		d (string): disclosures conditional ('x', '0', '1')
		model_type (string): Type of model ('100yr' or '500yr')

	Returns:
		None
	"""

	### Get outputs		
	coeffs = model.params.to_dict()
	pvalues = model.pvalues.to_dict()
	std_errors = model.std_errors.to_dict()
	ci_l = model.conf_int()['lower']
	ci_u = model.conf_int()['upper']

	### Store output in a dictionary
	model_dict = {}
	model_dict['model_label'] = 'attitude-%s_coastal-%s_disclosure-%s' %(a, c, d)
	model_dict['n_trans'] = model.nobs
	model_dict['n_props'] = model.entity_info.total
	
	if '100yr_dummy' in coeffs: 
		model_dict['100yr_coeff'] = coeffs['100yr_dummy']
		model_dict['100yr_se'] = std_errors['100yr_dummy']
		model_dict['100yr_ci_lower'] = ci_l['100yr_dummy']
		model_dict['100yr_ci_upper'] = ci_u['100yr_dummy']
		model_dict['100yr_pvalue'] = pvalues['100yr_dummy']

	if '500yr_dummy' in coeffs: 
		model_dict['500yr_coeff'] = coeffs['500yr_dummy']
		model_dict['500yr_se'] = std_errors['500yr_dummy']
		model_dict['500yr_ci_lower'] = ci_l['500yr_dummy']
		model_dict['500yr_ci_upper'] = ci_u['500yr_dummy']
		model_dict['500yr_pvalue'] = pvalues['500yr_dummy']
				
	### Initialize filename
	fn = '%s_attitude-%s_coastal-%s_disclosure-%s.pickle' %(model_type, a, c, d)

	### Initialize full path to pickle file
	model_pickle_uri = os.path.join(paths.outputs_dir, 'StatsModel_Pickles', fn)

	### Export model object to pickle
	with open(model_pickle_uri, 'wb') as handle:
	    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return None

