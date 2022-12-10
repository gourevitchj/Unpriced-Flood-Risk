### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: main.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import driver

def main():
	"""High-level function for defining user-specified inputs and executing 
		all other functions.

	Arguments:
		None

	Returns:
		None
	"""

	### Initialize dictionary specifying which functions to execute
	driver_args = {
		
		### Preprocessing functions
		'preprocessing': [
			'export_parquet2csvs',
			'join_firms2ztrax',	
			'clean_data',
			'get_countylevel_summarystats',
			'create_countygroups',
			],

		### Statistical functions
		'stats': [
			'fit_100yr_model',
			],

		### Postprocessing functions
		'postprocessing': [
			### Collect outputs from statistical models
			'collect_statsoutputs',
			
			### Calculate and summarize overvaluation
			'calculate_overvaluation',
			'summarize_countyoutputs',
			'combine_atriskproperties',

			### Monte Carlo simulation
			'montecarlo_simulation',
			'collect_montecarlo_outputs',

			### Miscellaneous postprocessing functions
			'get_censustract_outputs',
			'fiscalvulnerability_analysis',

			### Postprocessing functions to generate data to share with media
			'collect_keyoutputs',

			### Assume non-SFHA discount is the same as SFHA discount
			'calculate_overvaluation_nonsfhadiscount',
			'summarize_countyoutputs_nonsfhadiscount',
			'combine_atriskproperties_nonsfhadiscount',
			'get_censustract_outputs_nonsfhadiscount',
			
			### Assume discounts based on cross-sectional model
			'calculate_overvaluation_xsdiscount',
			'summarize_countyoutputs_xsdiscount',
			'combine_atriskproperties_xsdiscount',
			'get_censustract_outputs_xsdiscount',
			],

		### Figure functions
		'figures': [
			### Main text
			'overvaluation_maps', # Fig. 1
			'histograms_byfloodzone',	# Fig. 2
			'overvaluation_bypopulationgroups_v2', # Fig. 3
			'overvaluation_munifinance', # Fig. 4 (upper panel)
			'munifinance_demographics', #Fig. 4 (lower panel)

			### Supplemental Materials
			'ztrax_transactions_map', # Fig. S1
			'fmv_map', # Fig. S2
			'npv_map', # Fig. S3
			'firmupdates_map', # Fig. S4
			'yaleclimatesurvey_map', # Fig. S5						
			'npv_histograms', # Fig. S6
			'empiricaldiscount_combinedgroups', # Fig. S7
			'empiricaldiscount_combinedgroups_xsmodel', # Fig. S8
			'overvaluation_maps_dr7_low', # Fig. S9
			'overvaluation_maps_dr1_high', # Fig. S10
			'histograms_byfloodzone_nonsfhadiscount', # Fig. S11
			'histograms_byfloodzone_xsdiscount', # Fig. S12
			'overvaluation_bypopulationgroups_scatter', # Fig. S13
			'overvaluation_bypopulationgroups_v2_nonsfhadiscount', # Fig. S14
			'overvaluation_bypopulationgroups_v2_xsdiscount', # Fig. S15
			'uncertainty_plots', # Fig. S16
			'overvaluation_rankedbystate', # Fig. S17
			]
		}

	### Call execute function within driver.py
	driver.execute(driver_args)

if __name__ == '__main__':
	main()