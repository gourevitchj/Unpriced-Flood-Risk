### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: paths.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import os

### Initialize path to Davis-Penn directory
stem = os.path.dirname(os.path.dirname(
		os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

### Initialize path to data directory and project directory
data_dir = os.path.join(stem, 'Data')
proj_dir = os.path.join(stem, 'Projects/Property_Markets')

### Initialize path to ZTRAX database
ztrax_parquet_dir = os.path.join(data_dir, 'PLACES/ParquetFiles_byState')

### Initialize path to ZTRAX CSV directory
ztrax_orig_csv_dir = os.path.join(data_dir, 'PLACES/CSVs_Original')

### Initialize path to ZTRAX CSV file directory, joined with FIRMs
ztrax_joinedfirms_csv_dir = os.path.join(data_dir, 'PLACES/CSVs_JoinedFIRMs')

### Initialize path to ZTRAX CSV file directory, fully cleaned
ztrax_cleaned_csv_dir = os.path.join(data_dir, 'PLACES/CSVs_Clean')

### Initialize path to ZTRAX shapefile directory, fully cleaned
ztrax_cleaned_shp_dir = os.path.join(data_dir, 'PLACES/Shapefiles_Clean')

### Initialize path to historical FIRMs directory
historicalfirms_dir = os.path.join(
	data_dir, 'Historical_FIRMs/State_Shapefiles')

### Initialize path to CSV with DFIRM effective dates
firm_effdate_csv_uri = os.path.join(
	data_dir, 'Historical_FIRMs/EffDates_byTract.csv')

### Initialize path to CSV with FHFA HPI for 3-digit zip codes
fhfa_hpi_csv_uri = os.path.join(data_dir, 'FHFA_HPI/HPI_AT_3zip.csv')

### Initialize path to CSV with counties sample size
countystats_csv_uri = os.path.join(data_dir, 'PLACES/county_stats.csv')

### Initialize path to 2019 DFIRMs directory
dfirms_dir = os.path.join(data_dir, 'NFHL/State_Shapefiles_fromJo')

### Initialize path to coastal counties CSV file
coastalcounties_csv_uri = os.path.join(
	data_dir, 'Boundaries/Counties/Coastal_Counties/coastline-counties-list.csv')

### Initialize path to Yale climate survey CSV file
climatesurvey_csv_uri = os.path.join(data_dir, 'Yale_ClimateSurvey/data.csv')

### Initialize path to state-level disclosure laws CSV file
disclosurelaws_csv_uri = os.path.join(
	data_dir, 'State_DisclosureLaws/state_disclosure_laws.csv')

### Initialize path to CSV file to store county groupings
countygroups_csv_uri = os.path.join(proj_dir, 'Outputs/County_Groups.csv')

### Initialize path to CSV file containing coefficient estimates from XS model
xs_coeffs_csv_uri = os.path.join(proj_dir, 'Outputs/XS_Coeffs_fromAdam.csv')

### Initialize path to CSV file with ACS data
acs_csv_uri = os.path.join(data_dir, 'ACS/Tracts/acs2019_extract.csv')

### Initialize path to CSV file with municipal finance data
munifinance_csv_uri = os.path.join(
	data_dir, 'Gov_Finance_Database/MunicipalData.csv')

### Initialize paths to census tract, county, and state shapefiles
tracts_shp_uri = os.path.join(
	data_dir, 'Boundaries/Tracts/2019/cb_2019_us_tract_500k_lower48_proj.shp')

counties_shp_uri = os.path.join(
	data_dir, 'Boundaries/Counties/cb_2020_us_county_500k_lower48_proj.shp')

states_shp_uri = os.path.join(
	data_dir, 'Boundaries/States/cb_2018_us_state_20m_lower48_proj.shp')

### Initialize path to stats directory
outputs_dir = os.path.join(proj_dir, 'Outputs')

### Initialize path to figures directory
figures_dir = os.path.join(proj_dir, 'Figures')

