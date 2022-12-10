### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: preprocessing.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow.parquet as pq
import statsmodels.api as sm

import reference, paths

def export_parquet2csvs():

	"""Split parquet files in county-level CSV files.

	Arguments:
		None

	Returns:
		None
	"""

	### Iterate through files in ZTRAX parquet directory
	for f in sorted(os.listdir(paths.ztrax_parquet_dir)):

		### If filename has parquet file extension...
		if f[-4:] == '.pqt':
			print('\t\t%s' %f)

			### Initialize path to parquet file
			parquet_file_uri = os.path.join(paths.ztrax_parquet_dir, f)

			### Read ZTRAX parquet file using pyarrow.parquet
			df = pq.read_table(parquet_file_uri)

			### Convert table to Pandas dataframe
			df = df.to_pandas()

			### Convert Pandas dataframe to Geopandas geodataframe based on
			### x and y coordinates for building location
			points_geom = gpd.points_from_xy(df['x_bld'], df['y_bld'])
			df = gpd.GeoDataFrame(df, geometry=points_geom)

			### Subset columns
			cols = [
				### General property ID information
				'fips', 'zip_id_2017', 'tract_id_2016', 'pid', 'geometry',

				### Sale information
				'sid', 'year', 'date', 'price', 

				### Appraisal information
				'val_t_za', 'val_yr_za',
				 
				### Year built
				'bld_yr',
				]

			### Append all EAL columns to columns of interest list
			for c in df.columns:
				if 'fld_eal' in c:
					cols.append(c)

			### Check that columns of interest are actually present in dataframe
			for c in cols:
				if c not in df.columns:
					print('ERROR: %s IS NOT PRESENT...QUIT APPLICATION' %c)
					quit()

			### Subset dataframe to columns of interest
			df = df[cols]

			### Iterate through counties in state-level dataframe
			for fips in sorted(df['fips'].unique()):
				print('\t\t\t%s' %fips)

				### Subset dataframe to individual county
				df_fips = df[df['fips']==fips]

				### Export to CSV
				csv_fn = '%s.csv' %fips
				csv_uri = os.path.join(paths.ztrax_orig_csv_dir, csv_fn)
				df_fips.to_csv(csv_uri, index=False)

	return None


def join_firms2ztrax():

	"""Join ZTRAX transactions with historical Flood Insurance Rate Maps.

	Arguments:
		None

	Returns:
		None
	"""

	### Initialize EPSG coordinate system
	epsg = 5070

	### Get list of counties with ZTRAX data
	fn_list = os.listdir(paths.ztrax_orig_csv_dir)
	
	### Initialize empty list to store CSV URIs
	csv_uri_list = []

	### Iterate through filenames in filename list
	for fn in fn_list:

		### Append CSV to list
		csv_uri = os.path.join(paths.ztrax_orig_csv_dir, fn)
		csv_uri_list.append(csv_uri)

	### Initialize list of state FIPS codes
	state_fips_list = reference.state_fips_dict.keys()

	### Iterate through states
	for state_fips in sorted(state_fips_list):
		
		### Get state name from state FIPS code
		state_name = reference.state_fips_dict[state_fips]

		### Print state FIPS code and name
		print('\t\t%s: %s' %(state_fips, state_name))

		### Initialize path to historical FIRM shapefile
		q3_shp_uri = os.path.join(
			paths.historicalfirms_dir, 'Q3_FIPS_%s.shp' %state_fips)

		### Initialize path to 2019 SFHA shapefile
		nfhl_shp_uri = os.path.join(
			paths.dfirms_dir, 'NFHL19_FIPS_%s.shp' %state_fips)

		### Read shapefiles to GeoPandas dataframes
		df_q3 = gpd.read_file(q3_shp_uri)
		df_nfhl = gpd.read_file(nfhl_shp_uri)

		### Subset FIRM dataframe columns
		q3_cols = ['Q3_CHAN', 'geometry']
		df_q3 = df_q3[q3_cols]
		
		nfhl_cols = ['NFHL19_F', 'NFHL19_Z', 'geometry']
		df_nfhl = df_nfhl[nfhl_cols]

		### Rename FIRM dataframe columns
		df_q3 = df_q3.rename(columns={'Q3_CHAN': 'q3_floodzone'})
		df_nfhl = df_nfhl.rename(columns={'NFHL19_Z': 'dfirm_floodzone_cat'})
		df_nfhl = df_nfhl.rename(columns={'NFHL19_F': 'dfirm_floodzone_code'})

		### Merge the two different flood zone columns in the NFHL dataframe
		df_nfhl['dfirm_floodzone'] = np.where(
			df_nfhl['dfirm_floodzone_cat'].isna(),
			df_nfhl['dfirm_floodzone_code'], df_nfhl['dfirm_floodzone_cat'])

		### Set coordinate system for flood layers
		df_q3 = df_q3.to_crs(epsg=epsg)
		df_nfhl = df_nfhl.to_crs(epsg=epsg)

		### Get list of ZTRAX CSV files for state
		state_csv_uri_list = []
		for csv_uri in csv_uri_list:
			
			### Get county FIPS code
			cnty_fips = os.path.basename(csv_uri)[:-4]
			
			### If first two digits of county FIPS code match state FIPS code...
			if cnty_fips[:2] == state_fips:
				### Append CSV URI to list
				state_csv_uri_list.append(csv_uri)

		### Iterate through counties in state CSV URI list
		for csv_uri in sorted(state_csv_uri_list):
			
			### Get county FIPS code
			cnty_fips = os.path.basename(csv_uri)[:-4]
			print('\t\t\t%s' %cnty_fips)

			### Read county-level ZTRAX CSV to Pandas dataframe
			df = pd.read_csv(csv_uri)

			### Convert dataframe to to geodataframe
			df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])
			df = gpd.GeoDataFrame(df, geometry='geometry')

			### Set coordinate system
			df = df.set_crs(epsg=epsg)
			
			### Spatially join ZTRAX and FIRM layers using 'within'
			df = gpd.sjoin(df, df_q3, predicate='within', how='left')
			df = df.drop(['index_right'], axis=1)
			
			df = gpd.sjoin(df, df_nfhl, predicate='within', how='left')
			df = df.drop(['index_right'], axis=1)
			
			### Create URI for exported CSV
			csv_fn = '%s.csv' %cnty_fips
			fips_export_csv_uri = os.path.join(
				paths.ztrax_joinedfirms_csv_dir, csv_fn)

			### Export dataframe back to CSV
			df.to_csv(fips_export_csv_uri, index=False)

	return None


def clean_data():
	
	"""Clean data.

	Arguments:
		None

	Returns:
		None
	"""

	### Get list of available counties
	fn_list = os.listdir(paths.ztrax_joinedfirms_csv_dir)
	
	### Initialize empty list to store CSV URIs
	csv_uri_list = []
	
	### Iterate through filenames in filename list
	for fn in sorted(fn_list):
		
		### Append CSV to list
		csv_uri = os.path.join(paths.ztrax_joinedfirms_csv_dir, fn)
		csv_uri_list.append(csv_uri)

	### Read FIRM effective dates CSV file to Pandas dataframe
	df_effdates = pd.read_csv(paths.firm_effdate_csv_uri)

	################### Preprocess FIRM effective dates data ###################
	
	### Change effective date column type to datetime and rename as 'eff_date'
	df_effdates['eff_date'] = pd.to_datetime(
		df_effdates['Effective_date_first_DFIRM'])
	
	### Subset dataframe to census tract ID and effective date
	df_effdates = df_effdates[['GEOID', 'eff_date']]

	### Rename 'GEOID' column as 'geoid'
	df_effdates = df_effdates.rename(columns={'GEOID': 'geoid'})

	### Change geoid column type to string
	df_effdates['geoid'] = df_effdates['geoid'].astype(str)
	
	### Where length of geoid column is 10, add a leading '0'
	df_effdates['geoid'] = np.where(
		df_effdates['geoid'].str.len()==10, 
		'0'+df_effdates['geoid'], df_effdates['geoid'])

	############################################################################

	### Iterate through counties in CSV URI list
	for fips_csv_uri in csv_uri_list:

		### Get county FIPS code
		fips = os.path.basename(fips_csv_uri)[:-4]
		print('\t\t%s' %fips)

		### Read county-level ZTRAX CSV to Pandas dataframe
		df = pd.read_csv(fips_csv_uri, low_memory=False)

		### Drop columns 'dfirm_floodzone_code' and 'dfirm_floodzone_cat'
		cols = ['dfirm_floodzone_code', 'dfirm_floodzone_cat']
		for c in cols:
			if c in df.columns:
				df = df.drop(c, axis=1) 

		### Rename census tract and zip code columns
		df = df.rename(columns={
			'tract_id_2016': 'tract',
			'zip_id_2017': 'zip'})

		################### Create census tract 'geoid' field ##################
		
		### If tract is NaN, impute it with county mode
		df['tract'] = np.where(
			df['tract'].isna(), df['tract'].mode()[0], df['tract'])

		### Change census tract ID field type to string
		df['tract'] = df['tract'].astype(int).astype(str)
		
		### If length of string is 3, add three leading '0'
		df['tract'] = np.where(
			df['tract'].str.len()==3, 
			'000' + df['tract'], df['tract'])
		
		### If length of string is 4, add two leading '0'
		df['tract'] = np.where(
			df['tract'].str.len()==4, 
			'00'  + df['tract'], df['tract'])
		
		### If length of string is 5, add one leading '0'
		df['tract'] = np.where(
			df['tract'].str.len()==5, 
			'0'   + df['tract'], df['tract'])

		### Change county FIPS field type to string
		df['fips'] = df['fips'].astype(int).astype(str)

		### If length of string is 4, add one leading '0'
		df['fips'] = np.where(
			df['fips'].str.len()==4, 
			'0' + df['fips'], df['fips'])

		### Concatenate county fips ID and census tract ID fields
		df['geoid'] = df['fips'] + df['tract']

		##################### Rename flood zone categories #####################
		
		### Replace Q3 floodzone codes with intuitive labels
		df['q3_floodzone'] = df['q3_floodzone'].map(reference.floodzone_dict)
		
		### If Q3 floodzone has no label, then label it as unmapped
		df['q3_floodzone'] = np.where(
			df['q3_floodzone'].isin(['100yr', '500yr', 'outside', 'unmapped']),
			df['q3_floodzone'], 'unmapped')

		### Replace DFIRM floodzone codes with intuitive labels
		df['dfirm_floodzone'] = df['dfirm_floodzone'].map(reference.floodzone_dict)
		
		### If DFIRM floodzone has no label, then label it as unmapped
		df['dfirm_floodzone'] = np.where(
			df['dfirm_floodzone'].isin(['100yr', '500yr', 'outside', 'unmapped']),
			df['dfirm_floodzone'], 'unmapped')

		###### Assign transaction FIRM category based on effective dates #######
		
		### Change 'date' column type to datetime
		df['date'] = pd.to_datetime(df['date'])
		
		### Initialize empty columns to store data
		df['flood_zone'] = np.nan
		df['eff_date'] = np.nan
		df['before_after_update'] = np.nan

		### Get effective date column from effective dates dataframe
		eff_dates = df_effdates['eff_date']

		### Iterate through unique census tract IDs
		for geoid in df['geoid'].unique():
			
			### Get specific effective date for specific census tract
			eff_date = eff_dates[df_effdates['geoid']==geoid]

			if len(eff_date)!=1:
				print('ERRRRRRRRROR')
				quit()

			### If length of effective date series is one...
			if len(eff_date)==1:

				### Get singular value from series
				eff_date = eff_date.values[0]		
			
				### If sale occurred before effective date, assign Q3 floodzone
				df['flood_zone'] = np.where(
					(df['geoid']==geoid) & (df['date'] < eff_date),
					df['q3_floodzone'], df['flood_zone'])

				### If sale occurred after effective date, assign DFIRM floodzone
				df['flood_zone'] = np.where(
					(df['geoid']==geoid) & (df['date'] >= eff_date),
					df['dfirm_floodzone'], df['flood_zone'])

				### If property was not transacted, assign DRIRM floodzone
				df['flood_zone'] = np.where(df['date'].isna(), 
					df['dfirm_floodzone'], df['flood_zone'])

				### Assign effective date to each transaction
				df['eff_date'] = np.where(
					df['geoid']==geoid, str(eff_date)[:10], df['eff_date'])

				### If transaction occurred before update, label 'before'
				df['before_after_update'] = np.where(df['date'] < eff_date,
					'before', df['before_after_update'])

				### If transaction occurred before update, label 'after'
				df['before_after_update'] = np.where(df['date'] >= eff_date,
					'after', df['before_after_update'])

		############# Get current fair market value using FHFA HPI #############

		### Convert 'zip' column type to string
		df['zip'] = df['zip'].astype(str)

		### If length of zip code is four, add a leading zero
		df['zip'] = np.where(
			df['zip'].str.len()==4, 
			'0' + df['zip'], df['zip'])

		### Get three digits of zip code
		df['zip_3digit'] = df['zip'].str[:3]

		### Read FHFA HPI CSV file to Pandas DataFrame
		df_hpi = pd.read_csv(paths.fhfa_hpi_csv_uri)

		### Convert 'threedigit_zip' column type to string
		df_hpi['threedigit_zip'] = df_hpi['threedigit_zip'].astype(str)

		### If length of three digit zip code is two, add a leading zero
		df_hpi['threedigit_zip'] = np.where(
			df_hpi['threedigit_zip'].str.len()==2, 
			'0' + df_hpi['threedigit_zip'], df_hpi['threedigit_zip'])

		### Get month of transaction
		df['month'] = pd.DatetimeIndex(df['date']).month

		### Get quarter of transaction
		df['quarter'] = 1
		df['quarter'] = np.where(df['month']>3, 2, df['quarter'])
		df['quarter'] = np.where(df['month']>6, 3, df['quarter'])
		df['quarter'] = np.where(df['month']>9, 4, df['quarter'])

		### Create current fair market value column in dataframe
		df['price_hpiadjusted'] = np.nan
		df['val_hpiadjusted'] = np.nan
		df['fmv'] = np.nan

		### Iterate through unique zip codes in dataframe
		for z in df['zip_3digit'].unique():

			### Get 2021 Q4 data
			q4_2021_hpi = df_hpi['index_nsa'][(df_hpi['threedigit_zip']==z) &
											  (df_hpi['year']==2021) &
											  (df_hpi['quarter']==4)]

			### If zip code is NaN and HPI is unknown...
			if len(q4_2021_hpi) == 0:
				q4_2021_hpi = df_hpi['index_nsa'][
					(df_hpi['threedigit_zip'].isin(df['zip_3digit'])) &
					(df_hpi['year']==2021) &
					(df_hpi['quarter']==4)].mean()					

			### Convert Series to float	
			q4_2021_hpi = float(q4_2021_hpi)

			### Iterate through unique non-nan sale years in dataframe
			for yr in df['year'].dropna().unique():

				### Iterate through unique non-nan quarters in dataframe
				for q in df['quarter'].dropna().unique():

					### Get year, quarter HPI data
					hpi = df_hpi['index_nsa'][(df_hpi['threedigit_zip']==z) & 
								 			  (df_hpi['year']==yr) &
											  (df_hpi['quarter']==q)]
					
					### If zip code is NaN and HPI is unknown...
					if len(hpi) == 0:
						hpi = df_hpi['index_nsa'][
							(df_hpi['threedigit_zip'].isin(df['zip_3digit'])) &
							(df_hpi['year']==yr) &
							(df_hpi['quarter']==q)].mean()		

					### Convert Series to float
					hpi = float(hpi)

					### Get difference between 2021 HPI and transaction HPI
					hpi_scalar = (q4_2021_hpi - hpi) / hpi

					### Calculate FMV base applying HPI scalar to price
					df['price_hpiadjusted'] = np.where(
						(df['zip_3digit']==z) &
						(df['year']==yr) &
						(df['quarter']==q),

						df['price'] * (1 + hpi_scalar),
						df['price_hpiadjusted'])

			### Iterate through unique non-nan appraisal years in dataframe
			for yr in df['val_yr_za'].dropna().unique():

				### Get average HPI data for years
				hpi = df_hpi['index_nsa'][(df_hpi['threedigit_zip']==z) & 
								 			  (df_hpi['year']==yr)]

				### If zip code is NaN and HPI is unknown...
				if len(hpi) == 0:
					hpi = df_hpi['index_nsa'][
						(df_hpi['threedigit_zip'].isin(df['zip_3digit'])) &
						(df_hpi['year']==yr)].mean()	

				### Convert Series to float
				hpi = float(hpi.mean())

				### Get difference between 2021 HPI and transaction HPI
				hpi_scalar = (q4_2021_hpi - hpi) / hpi

				### Calculate FMV by applying HPI scalar to price
				df['val_hpiadjusted'] = np.where(
						(df['zip_3digit']==z) & (df['val_yr_za']==yr),
						df['val_t_za'] * (1 + hpi_scalar),
						df['val_hpiadjusted'])	

		### Fill 'fmv' column with HPI adjusted prices
		df['fmv'] = np.where(
			~df['price_hpiadjusted'].isna(), 
			df['price_hpiadjusted'], 
			df['fmv'])		

		### Fit linear model between HPI adjusted prices and HPI adjusted 
		### appraised values
		xy = df[['fmv', 'val_hpiadjusted']].dropna()
		
		if len(xy) > 1:
			model = sm.OLS(xy['fmv'], xy['val_hpiadjusted']).fit()
			coeff = model.params['val_hpiadjusted']

		else:
			coeff = 1

		### Fill 'fmv' column with HPI adjusted appraised values, further
		### adjusted for difference between appraised values and sale prices
		df['fmv'] = np.where(
			df['price_hpiadjusted'].isna() & (~df['val_hpiadjusted'].isna()), 
			df['val_hpiadjusted'] * coeff, 
			df['fmv'])			


		################### Adjust sale prices for inflation ###################

		### Create price columns
		df['price_nominal'] = df['price'].copy() # Unadjusted price
		df['price_real'] = np.nan # Adjusted price

		### Iterate through unique years
		for yr in df['year'].dropna().unique():
			
			if yr >= 1990:
				### Get CPI scalar for specific year
				cpi_scalar = reference.cpi_dict[str(int(yr))]
			
				### For specific year, apply specific CPI scalar
				df['price_real'] = np.where(df['year']==yr, 
					df['price_nominal'] * cpi_scalar, df['price_real'])

		### Delete 'price' column
		df = df.drop('price', axis=1)
		
		####################################################################

		### Add column for property age
		df['prop_age'] = df['year'] - df['bld_yr']

		### Reorder columns
		cols = ['sid', 'pid', 'geometry', 'geoid', 'tract', 'zip',
				'zip_3digit', 'fips', 'date', 'year', 'month', 'quarter', 
				'price_nominal', 'price_real', 'price_hpiadjusted', 'fmv', 
				'val_hpiadjusted', 'val_t_za', 'val_yr_za',	'bld_yr',
				'q3_floodzone', 'dfirm_floodzone', 'flood_zone', 
				'before_after_update', 'eff_date', 'prop_age']
		for c in df.columns:
			if c not in cols:
				cols.append(c)

		df = df[cols]

		### Export dataframe back to CSV
		fn = os.path.basename(fips_csv_uri)
		fips_clean_csv_uri = os.path.join(paths.ztrax_cleaned_csv_dir, fn)
		df.to_csv(fips_clean_csv_uri, index=False)

	return None


def get_countylevel_summarystats():

	"""Get county-level summary statistics.

	Arguments:
		None

	Returns:
		None
	"""

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
	rows_list = []

	### Iterate through list of county CSV files
	for csv_uri in csv_uri_list[:10]:

		### Get county FIPS code 
		cnty_fips = os.path.basename(csv_uri)[:5]
		
		### Get state FIPS code
		state_fips = cnty_fips[:2]
		
		### Print county FIPS code
		print('\t\t%s' %cnty_fips)

		### Read county CSV file to Pandas DataFrame
		df = pd.read_csv(csv_uri, low_memory=False)

		### Sort dataframe by 'pid' and 'year'
		df = df.sort_values(['pid', 'date']).reset_index(drop=True)

		### Only include transactions with price greater than zero
		df = df[df['price_nominal']>0]

		### Remove properties with only one transactions
		df_duplicated = df[df.duplicated(subset='pid', keep=False)]
		df = df[df['pid'].isin(df_duplicated['pid'].unique())]

		### Get number of unique properties
		n_prop = len(df['pid'].unique())

		### Get number of transactions
		n_trans = len(df)

		### Get dataframes with first and last transactions
		df_first = df.drop_duplicates(subset='pid', keep='first')
		df_last = df.drop_duplicates(subset='pid', keep='last')

		### Subset first and last dataframe to columns of interest
		df_first = df_first[['pid', 'flood_zone']]
		df_last = df_last[['pid', 'flood_zone']]

		### Rename flood zone columns
		df_first = df_first.rename(columns={'flood_zone': 'fz_first'})
		df_last = df_last.rename(columns={'flood_zone': 'fz_last'})

		### Merge first and last transaction dataframes
		df = df_first.merge(df_last, on='pid')

		### If first transaction was unmapped...
		n_unmapped_to_unmapped = len(
			df[(df['fz_first']=='unmapped') & 
			  (df['fz_last']=='unmapped')])
		n_unmapped_to_outside = len(
			df[(df['fz_first']=='unmapped') & 
			   (df['fz_last']=='outside')])
		n_unmapped_to_500 = len(
			df[(df['fz_first']=='unmapped') & 
			   (df['fz_last']=='500yr')])
		n_unmapped_to_100 = len(
			df[(df['fz_first']=='unmapped') & 
			   (df['fz_last']=='100yr')])

		### If first transaction was outside flood zone...
		n_outside_to_unmapped = len(
			df[(df['fz_first']=='outside') & 
			   (df['fz_last']=='unmapped')])
		n_outside_to_outside = len(
			df[(df['fz_first']=='outside') & 
			   (df['fz_last']=='outside')])
		n_outside_to_500 = len(
			df[(df['fz_first']=='outside') & 
			   (df['fz_last']=='500yr')])
		n_outside_to_100 = len(
			df[(df['fz_first']=='outside') & 
			   (df['fz_last']=='100yr')])

		### If first transaction was in 500-yr flood zone...
		n_500_to_unmapped = len(
			df[(df['fz_first']=='500yr') & 
			   (df['fz_last']=='unmapped')])
		n_500_to_outside = len(
			df[(df['fz_first']=='500yr') & 
			   (df['fz_last']=='outside')])
		n_500_to_500 = len(
			df[(df['fz_first']=='500yr') & 
			   (df['fz_last']=='500yr')])
		n_500_to_100 = len(
			df[(df['fz_first']=='500yr') & 
			   (df['fz_last']=='100yr')])

		### If first transaction was in 100-yr flood zone...
		n_100_to_unmapped = len(
			df[(df['fz_first']=='100yr') & 
			   (df['fz_last']=='unmapped')])
		n_100_to_outside = len(
			df[(df['fz_first']=='100yr') & 
			   (df['fz_last']=='outside')])
		n_100_to_500 = len(
			df[(df['fz_first']=='100yr') & 
			   (df['fz_last']=='500yr')])
		n_100_to_100 = len(
			df[(df['fz_first']=='100yr') & 
			   (df['fz_last']=='100yr')])

		### Store data in row_dict
		row_dict = {
			'state_fips': state_fips,
			'cnty_fips': cnty_fips,
			'n_prop': n_prop,
			'n_trans': n_trans,

			'n_prop_unmapped_to_unmapped': n_unmapped_to_unmapped,
			'n_prop_unmapped_to_outside': n_unmapped_to_outside,
			'n_prop_unmapped_to_500': n_unmapped_to_500,
			'n_prop_unmapped_to_100': n_unmapped_to_100,

			'n_prop_outside_to_outside': n_outside_to_outside,
			'n_prop_outside_to_500': n_outside_to_500,
			'n_prop_outside_to_100': n_outside_to_100,

			'n_prop_500_to_outside': n_500_to_outside,
			'n_prop_500_to_500': n_500_to_500,
			'n_prop_500_to_100': n_500_to_100,

			'n_prop_100_to_outside': n_100_to_outside,
			'n_prop_100_to_500': n_100_to_500,
			'n_prop_100_to_100': n_100_to_100,
			}

		### Append row_dict to rows_list
		rows_list.append(row_dict)

	### Convert row list to dataframe
	df_final = pd.DataFrame(rows_list)

	### Export dataframe to CSV
	df_final.to_csv(paths.countystats_csv_uri, index=False)

	return None


def create_countygroups():

	"""Group counties by characteristics.

	Arguments:
		None

	Returns:
		None
	"""

	#################### Prepare flood risk disclosure data ####################
	
	### Initialize path to flood risk disclosure CSV file
	disclosure_csv_uri = paths.disclosurelaws_csv_uri

	### Read flood risk disclosure CSV file to Pandas dataframe
	df1 = pd.read_csv(disclosure_csv_uri)

	### Rename columns
	df1 = df1.rename(columns={
		'FIPS_ST': 'state_fips',
		'Floodplain': 'floodplain_disclosure',
	  	'Flood damage': 'damage_disclosure',
	  	'Flood ins carried or required': 'insurance_disclosure'})
	
	### Drop unneeded columns
	df1 = df1.drop(['Notes', 'State'], axis=1)

	### Change state FIPS type to integer
	df1['state_fips'] = df1['state_fips'].astype(int)

	### Create binary variables for each form of disclosure requirement
	df1['floodplain_disclosure'] = np.where(
		df1['floodplain_disclosure'] == 1, 1, 0)
	df1['damage_disclosure'] = np.where(
		df1['damage_disclosure'] == 1, 1, 0)
	df1['insurance_disclosure'] = np.where(
		df1['insurance_disclosure'] == 1, 1, 0)

	### Create aggregate disclosure score
	df1['disclosure_score'] = (
		df1['floodplain_disclosure'] + 
		df1['damage_disclosure'] + 
		df1['insurance_disclosure']
		)

	### Create binary disclosure variable
	df1['disclosure_binary'] = np.where(df1['disclosure_score'] > 0, 1, 0)

	#################### Prepare coastal counties dataframe ####################

	### Initialize path to coastal counties CSV file
	coastal_csv_uri = paths.coastalcounties_csv_uri

	### Read coastal counties CSV file to Pandas dataframe
	df2 = pd.read_csv(coastal_csv_uri)

	### Rename columns
	df2 = df2.rename(columns={'FIPS': 'fips',})
	
	### Drop unneeded columns
	df2 = df2.drop(['COASTLINE_REGION', 'STATE_FIPS', 'COUNTY_FIPS',
					'COUNTY_NAME', 'STATE_NAME'], axis=1)

	### Create binary coastal variable
	df2['coastal_binary'] = 1

	#################### Prepare climate attitudes dataframe ###################
	
	### Initialize path to climate attitudes CSV file
	attitudes_csv_uri = paths.climatesurvey_csv_uri

	### Read climate attitudes CSV file to Pandas dataframe	
	df3 = pd.read_csv(attitudes_csv_uri)

	### Subset data to only county-level data
	df3 = df3[df3['GeoType']=='County']

	### Subset to columns of interest
	df3 = df3[['GEOID', 'worried', 'happening', 'personal']]

	### Rename columns
	df3 = df3.rename(columns={
		'GEOID': 'fips',
		'worried': 'perc_worried',
		'happening': 'perc_happening',
		'personal': 'perc_personal',
		})

	### Get state FIPS code
	df3['state_fips'] = df3['fips'].astype(str).str[:-3]
	
	### Convert state FIPS code type to integer
	df3['state_fips'] = df3['state_fips'].astype(int)

	### Create percent worried quartiles column
	df3['perc_worried_quartile'] = np.nan

	### Fill in percent worried quartiles data
	df3['perc_worried_quartile'] = np.where(
		df3['perc_worried'] < np.percentile(df3['perc_worried'], 25),
		1, df3['perc_worried_quartile'])

	df3['perc_worried_quartile'] = np.where(
		(df3['perc_worried'] >= np.percentile(df3['perc_worried'], 25)) &
		(df3['perc_worried'] <  np.percentile(df3['perc_worried'], 50)),
		2, df3['perc_worried_quartile'])

	df3['perc_worried_quartile'] = np.where(
		(df3['perc_worried'] >= np.percentile(df3['perc_worried'], 50)) &
		(df3['perc_worried'] <  np.percentile(df3['perc_worried'], 75)),
		3, df3['perc_worried_quartile'])

	df3['perc_worried_quartile'] = np.where(
		df3['perc_worried'] >= np.percentile(df3['perc_worried'], 75),
		4, df3['perc_worried_quartile'])

	df3['perc_worried_binary'] = np.where(df3['perc_worried_quartile']>2, 1, 0)

	### Drop percent worried column
	df3 = df3.drop('perc_worried', axis=1)

	### Create percent happening quartiles column
	df3['perc_happening_quartile'] = np.nan

	### Fill in percent happening quartiles data
	df3['perc_happening_quartile'] = np.where(
		df3['perc_happening'] < np.percentile(df3['perc_happening'], 25),
		1, df3['perc_happening_quartile'])

	df3['perc_happening_quartile'] = np.where(
		(df3['perc_happening'] >= np.percentile(df3['perc_happening'], 25)) &
		(df3['perc_happening'] <  np.percentile(df3['perc_happening'], 50)),
		2, df3['perc_happening_quartile'])

	df3['perc_happening_quartile'] = np.where(
		(df3['perc_happening'] >= np.percentile(df3['perc_happening'], 50)) &
		(df3['perc_happening'] <  np.percentile(df3['perc_happening'], 75)),
		3, df3['perc_happening_quartile'])

	df3['perc_happening_quartile'] = np.where(
		df3['perc_happening'] >= np.percentile(df3['perc_happening'], 75),
		4, df3['perc_happening_quartile'])

	df3['perc_happening_binary'] = np.where(df3['perc_happening_quartile']>2, 1, 0)

	### Drop percent happening column
	df3 = df3.drop('perc_happening', axis=1)

	### Create percent happening quartiles column
	df3['perc_personal_quartile'] = np.nan

	### Fill in percent happening quartiles data
	df3['perc_personal_quartile'] = np.where(
		df3['perc_personal'] < np.percentile(df3['perc_personal'], 25),
		1, df3['perc_personal_quartile'])

	df3['perc_personal_quartile'] = np.where(
		(df3['perc_personal'] >= np.percentile(df3['perc_personal'], 25)) &
		(df3['perc_personal'] <  np.percentile(df3['perc_personal'], 50)),
		2, df3['perc_personal_quartile'])

	df3['perc_personal_quartile'] = np.where(
		(df3['perc_personal'] >= np.percentile(df3['perc_personal'], 50)) &
		(df3['perc_personal'] <  np.percentile(df3['perc_personal'], 75)),
		3, df3['perc_personal_quartile'])

	df3['perc_personal_quartile'] = np.where(
		df3['perc_personal'] >= np.percentile(df3['perc_personal'], 75),
		4, df3['perc_personal_quartile'])

	df3['perc_personal_binary'] = np.where(df3['perc_personal_quartile']>2, 1, 0)

	df3['perc_personal_90thperc'] = np.where(
		df3['perc_personal'] >= np.percentile(df3['perc_personal'], 90), 1, 0)

	### Drop percent happening column
	df3 = df3.drop('perc_personal', axis=1)

	############################# Merge and export #############################
	
	### Merge coastal data to climate attitudes data
	df = df3.merge(df2, on='fips', how='left')
	
	### In coastal binary field, fill NaN values with zero values
	df['coastal_binary'] = df['coastal_binary'].fillna(0)

	### Merge flood risk disclosure dataframe to final dataframe
	df = df.merge(df1, on='state_fips', how='left')

	### Initialize path to export file
	countygroups_csv_uri = os.path.join(paths.outputs_dir, 'County_Groups.csv')

	### Export final dataframe to CSV
	df.to_csv(countygroups_csv_uri, index=False)

	return None

