### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: driver.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Import packages
import utils

def execute(driver_args):

	"""Execute functions used for analysis.

	Arguments:
		driver_args (dict): dictionary specifying which functions should by run
	
	Returns:
		None
	"""

	############################## Preprocessing ###############################

	### If at least one preprocessing function is called...
	if len(driver_args['preprocessing']) > 0:
		print('Preprocessing input data...')

		### Import preprocessing module
		import preprocessing

		if 'export_parquet2csvs' in driver_args['preprocessing']:
			### Split parquet files into county-level CSV files
			print('\tSplitting parquet file into county-level CSV files')
			preprocessing.export_parquet2csvs()
			utils.send_textmessage(
				'Finished splitting parquet file into county-level CSV files')

		if 'join_firms2ztrax' in driver_args['preprocessing']:
			### Spatially join ZTRAX transactions with historical FIRMs
			print('\tJoining historical FIRMs to ZTRAX transactions')
			preprocessing.join_firms2ztrax()
			utils.send_textmessage(
				'Finished joining historical FIRMs to ZTRAX transactions')

		if 'clean_data' in driver_args['preprocessing']:
			### Clean data
			print('\tCleaning data')
			preprocessing.clean_data()
			utils.send_textmessage('Finished cleaning data')

		if 'get_countylevel_summarystats' in driver_args['preprocessing']:
			### Get counties sample size
			print('\tGetting county-level summary statistics')
			preprocessing.get_countylevel_summarystats()
			utils.send_textmessage(
				'Finished getting county-level summary statistics')

		if 'create_countygroups' in driver_args['preprocessing']:
			### Group counties by characteristics
			print('\tGrouping counties by characteristics')
			preprocessing.create_countygroups()


	################################## Stats ###################################

	### If at least one statistical function is called...
	if len(driver_args['stats']) > 0:
		print('Running statistical analysis...')

		### Import stats module
		import stats
		
		if 'fit_100yr_model' in driver_args['stats']:
			### Fit panel model
			print('\tFitting panel models to estimate 100-yr effect')
			model_type = '100yr'
			stats.execute(model_type)
			utils.send_textmessage(
				'Finished fitting panel models to estimate 100-yr effect')


	############################# Postprocessing ###############################

	### If at least one postprocessing function is called...
	if len(driver_args['postprocessing']) > 0:
		print('Executing postprocessing functions...')

		### Import postprocessing module
		import postprocessing

		if 'collect_statsoutputs' in driver_args['postprocessing']:
			### Collect hedonic model outputs
			print('\tCollecting hedonic model outputs')
			postprocessing.collect_statsoutputs()
			utils.send_textmessage('Finished collecting hedonic model outputs')

		if 'calculate_overvaluation' in driver_args['postprocessing']:
			### Calculate potential overvaluation
			print('\tCalculating property-level overvaluation')
			postprocessing.calculate_overvaluation()
			utils.send_textmessage('Finished calculating potential overvaluation')

		if 'calculate_overvaluation_nonsfhadiscount' in driver_args['postprocessing']:
			### Calculate potential overvaluation assuming non-SFHA discount
			print('\tCalculating property-level overvaluation assuming non-SFHA discount')
			postprocessing.calculate_overvaluation(assume_nonsfha_discount=True)
			utils.send_textmessage(
				'Finished calculating potential overvaluation assuming non-SFHA discount')

		if 'calculate_overvaluation_xsdiscount' in driver_args['postprocessing']:
			### Calculate potential overvaluation assuming XS discount
			print('\tCalculating property-level overvaluation assuming XS discount')
			postprocessing.calculate_overvaluation(assume_xs_discount=True)
			utils.send_textmessage(
				'Finished calculating potential overvaluation assuming XS discount')

		if 'summarize_countyoutputs' in driver_args['postprocessing']:
			### Summarize county-level outputs 
			print('\tSummarizing county-level overvaluation outputs')
			postprocessing.summarize_countyoutputs()
			utils.send_textmessage('Finished combining county-level outputs')

		if 'summarize_countyoutputs_nonsfhadiscount' in driver_args['postprocessing']:
			### Combine county-level outputs assuming non-SFHA discount
			print('\tCombining county-level outputs assuming non-SFHA discount')
			postprocessing.summarize_countyoutputs(assume_nonsfha_discount=True)
			utils.send_textmessage(
				'Finished combining county-level outputs assuming non-SFHA discount')

		if 'summarize_countyoutputs_xsdiscount' in driver_args['postprocessing']:
			### Combine county-level outputs assuming XS discount
			print('\tCombining county-level outputs assuming XS discount')
			postprocessing.summarize_countyoutputs(assume_xs_discount=True)
			utils.send_textmessage(
				'Finished combining county-level outputs assuming XS discount')

		if 'combine_atriskproperties' in driver_args['postprocessing']:
			### Combine at-risk properties
			print('\tCombining at-risk properties')
			postprocessing.combine_atriskproperties()
			utils.send_textmessage('Finished combining at-risk properties')

		if 'combine_atriskproperties_nonsfhadiscount' in driver_args['postprocessing']:
			### Combining at-risk properties assuming non-SFHA discount
			print('\tCombining at-risk properties assuming non-SFHA discount')
			postprocessing.combine_atriskproperties(assume_nonsfha_discount=True)
			utils.send_textmessage(
				'Finished combining at-risk properties assuming non-SFHA discount')

		if 'combine_atriskproperties_xsdiscount' in driver_args['postprocessing']:
			### Combine at-risk properties assuming XS discount
			print('\tCombining at-risk properties assuming XS discount')
			postprocessing.combine_atriskproperties(assume_xs_discount=True)
			utils.send_textmessage(
				'Finished combining at-risk properties assuming XS discount')

		if 'montecarlo_simulation' in driver_args['postprocessing']:
			### Run Monte Carlo simulation
			print('\tRunning Monte Carlo simulation')
			postprocessing.montecarlo_simulation()
			utils.send_textmessage('Finished Monte Carlo simulation')

		if 'collect_montecarlo_outputs' in driver_args['postprocessing']:
			### Collect Monte Carlo outputs
			print('\tCollecting Monte Carlo outputs')
			postprocessing.collect_montecarlo_outputs()
			utils.send_textmessage('Finished collecting Monte Carlo outputs')

		if 'get_censustract_outputs' in driver_args['postprocessing']:
			### Get census tract-level outputs
			print('\tGetting census tract-level outputs')
			postprocessing.get_censustract_outputs()
			utils.send_textmessage('Finished getting census tract-level outputs')

		if 'get_censustract_outputs_nonsfhadiscount' in driver_args['postprocessing']:
			### Get census tract-level outputs assuming non-SFHA discount
			print('\tGetting census tract-level outputs assuming non-SFHA discount')
			postprocessing.get_censustract_outputs(assume_nonsfha_discount=True)
			utils.send_textmessage(
				'Finished getting census tract-level outputs assuming non-SFHA discount')

		if 'get_censustract_outputs_xsdiscount' in driver_args['postprocessing']:
			### Get census tract-level outputs assuming XS discount
			print('\tGetting census tract-level outputs assuming XS discount')
			postprocessing.get_censustract_outputs(assume_xs_discount=True)
			utils.send_textmessage(
				'Finished getting census tract-level outputs assuming XS discount')

		if 'fiscalvulnerability_analysis' in driver_args['postprocessing']:
			### Run fiscal vulnerability analysis
			print('\tRunning fiscal vulnerability analysis')
			postprocessing.fiscalvulnerability_analysis()
			utils.send_textmessage('Finished fiscal vulnerability analysis')

		if 'collect_keyoutputs' in driver_args['postprocessing']:
			### Collect key outputs to share with media
			print('\tCollect key outputs to share with media')
			postprocessing.collect_keyoutputs()
			utils.send_textmessage('Finished collecting key outputs')


	################################# Figures ##################################

	### If at least one figure function is called...
	if len(driver_args['figures']) > 0:
		print('Generating figures...')

		### Import figures module
		import figures

		if 'empiricaldiscount_combinedgroups' in driver_args['figures']:
			### Generate point plots for flood zone discounts
			print('\tGenerating plot for flood zone discounts by combined groups')
			figures.empiricaldiscount_combinedgroups()

		if 'empiricaldiscount_combinedgroups_xsmodel' in driver_args['figures']:
			### Generate point plots for flood zone discounts for XS model
			print('\tGenerating plot for flood zone discounts by combined groups for XS model')
			figures.empiricaldiscount_combinedgroups_xsmodel()

		if 'overvaluation_maps' in driver_args['figures']:
			### Generate overvaluation maps
			print('\tGenerating overvaluation maps')
			figures.overvaluation_maps()

		if 'overvaluation_maps_dr1_high' in driver_args['figures']:
			### Generate overvaluation maps with 1% discount rate and high hazard scenario
			print(r'\tGenerating overvaluation maps with 1% discount rate and high hazard scenario')
			figures.overvaluation_maps(discount_rate=1, hazard_scenario='h')

		if 'overvaluation_maps_dr7_low' in driver_args['figures']:
			### Generate overvaluation maps with 7% discount rate and low hazard scenario
			print(r'\tGenerating overvaluation maps with 7% discount rate and low hazard scenario')
			figures.overvaluation_maps(discount_rate=7, hazard_scenario='l')

		if 'overvaluation_rankedbystate' in driver_args['figures']:
			### Generate figure for overvaluation ranked by state
			print('\tGenerating figure for overvaluation ranked by state')
			figures.overvaluation_rankedbystate()

		if 'overvaluation_bypopulationgroups_v2' in driver_args['figures']:
			### Generate figure for overvaluation by population groups
			print('\tGenerating figure for overvaluation by population groups')
			figures.overvaluation_bypopulationgroups_v2()

		if 'overvaluation_bypopulationgroups_v2_nonsfhadiscount' in driver_args['figures']:
			### Generate figure for overvaluation by population groups assuming non-SFHA discount
			print('\tGenerating figure for overvaluation by population groups assuming non-SFHA discount')
			figures.overvaluation_bypopulationgroups_v2(assume_nonsfha_discount=True)

		if 'overvaluation_bypopulationgroups_v2_xsdiscount' in driver_args['figures']:
			### Generate figure for overvaluation by population groups assuming XS discount
			print('\tGenerating figure for overvaluation by population groups assuming XS discount')
			figures.overvaluation_bypopulationgroups_v2(assume_xs_discount=True)

		if 'overvaluation_bypopulationgroups_scatter' in driver_args['figures']:
			### Generate figure for overvaluation by population groups as *scatter plot*
			print('\tGenerating figure for overvaluation by population groups as *scatter plot*')
			figures.overvaluation_bypopulationgroups_scatter()

		if 'overvaluation_munifinance' in driver_args['figures']:
			### Generate figure for overvaluation by municipal finance
			print('\tGenerating figure for overvaluation by municipal finance')
			figures.overvaluation_munifinance()
		
		if 'yaleclimatesurvey_map' in driver_args['figures']:
			### Generate Yale Climate Survey map
			print('\tGenerating Yale Climate Survey map')
			figures.yaleclimatesurvey_map()

		if 'firmupdates_map' in driver_args['figures']:
			### Generate FIRM updates map
			print('\tGenerating FIRM updates map')
			figures.firmupdates_map()

		if 'histograms_byfloodzone' in driver_args['figures']:
			### Generate histograms by flood zone
			print('\tGenerating histograms by flood zone')
			figures.histograms_byfloodzone()

		if 'uncertainty_plots' in driver_args['figures']:
			### Generate plots for uncertainty analysis
			print('\tGenerating plots for uncertainty analysis')
			figures.uncertainty_plots()

		if 'munifinance_demographics' in driver_args['figures']:
			### Generate plots for municipal finance demographics
			print('\tGenerating plots for municipal finance demographics')
			figures.munifinance_demographics()

		if 'fmv_map' in driver_args['figures']:
			### Generate map for median FMVs by county
			print('\tGenerating map for median FMVs by county')
			figures.fmv_map()

		if 'ztrax_transactions_map' in driver_args['figures']:
			### Generate map for number of transactions per parcel
			print('\tGenerating map for number of transactions per parcel')
			figures.ztrax_transactions_map()

		if 'npv_map' in driver_args['figures']:
			### Generate map for NPV of flood losses by county
			print('\tGenerating map for NPV of flood losses by county')
			figures.npv_map()

		if 'npv_histograms' in driver_args['figures']:
			### Generate histograms for NPV of flood losses by SFHA / non-SFHA
			print('\tGenerating histograms for NPV of flood losses by SFHA / non-SFHA')
			figures.npv_histograms()

		if 'histograms_byfloodzone_nonsfhadiscount' in driver_args['figures']:
			### Generate histograms by flood zone assuming non-SFHA discount
			print('\tGenerating histograms by flood zone assuming non-SFHA discount')
			figures.histograms_byfloodzone(assume_nonsfha_discount=True)

		if 'histograms_byfloodzone_xsdiscount' in driver_args['figures']:
			### Generate histograms by flood zone assuming XS discount
			print('\tGenerating histograms by flood zone assuming XS discount')
			figures.histograms_byfloodzone(assume_xs_discount=True)

	############################################################################

	return None

	