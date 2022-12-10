### Project name: Unpriced climate risk and potential consequences of overvaluation in US housing markets
### Script name: reference.py
### Created by: Jesse D. Gourevitch
### Language: Python v3.9
### Last updated: December 9, 2022

### Initialize dictionary of FIPS codes and state names
state_fips_dict = {
	'01': 'Alabama', 
	# '02': 'Alaska', # Missing First Street data 
	'04': 'Arizona', 
	'05': 'Arkansas', 
	'06': 'California', 
	'08': 'Colorado',
	'09': 'Connecticut', 
	'10': 'Delaware', 
	# '11': 'Washington D.C.', # Missing from ZTRAX data
	'12': 'Florida', 
	'13': 'Georgia', 
	# '15': 'Hawaii', # Missing First Street data 
	'16': 'Idaho', 
	'17': 'Illinois', 
	'18': 'Indiana', 
	'19': 'Iowa',
	'20': 'Kansas', 
	'21': 'Kentucky', 
	'22': 'Louisiana', 
	'23': 'Maine', 
	'24': 'Maryland', 
	'25': 'Massachusetts', 
	'26': 'Michigan', 
	'27': 'Minnesota', 
	'28': 'Mississippi', 
	'29': 'Missouri',
	'30': 'Montana', 
	'31': 'Nebraska', 
	'32': 'Nevada', 
	'33': 'New Hampshire', 
	'34': 'New Jersey', 
	'35': 'New Mexico', 
	'36': 'New York', 
	'37': 'North Carolina', 
	'38': 'North Dakota', 
	'39': 'Ohio',
	'40': 'Oklahoma', 
	'41': 'Oregon', 
	'42': 'Pennsylvania', 
	'44': 'Rhode Island', 
	'45': 'South Carolina', 
	'46': 'South Dakota', 
	'47': 'Tennessee', 
	'48': 'Texas', 
	'49': 'Utah',
	'50': 'Vermont',
	'51': 'Virginia', 
	'53': 'Washington', 
	'54': 'West Virginia', 
	'55': 'Wisconsin', 
	'56': 'Wyoming',
	# '72': 'Puerto Rico', # Missing First Street data 
	}

### Initialize dictionary with FIRM flood zone codes and classifications
floodzone_dict = {
	### 100-yr flood zone
	'100 Year Chance': '100yr',
	'FLOODWAY': '100yr',
	'A': '100yr',
	'AE': '100yr',
	'AH': '100yr',
	'AO': '100yr',
	'A99': '100yr',
	'V': '100yr',
	'VE': '100yr',
	
	### 500-yr flood zone
	'500 Year Chance': '500yr',
	'0.2 PCT ANNUAL CHANCE FLOOD HAZARD': '500yr',

	### Outside flood zone
	'AREA OF MINIMAL FLOOD HAZARD': 'outside',
	'Outside Floodplain': 'outside',

	### Not mapped
	'Not Mapped': 'unmapped',
	'D': 'unmapped',
	}

### Initialize CPI conversions values for each year based on:
### https://www.bls.gov/data/inflation_calculator.htm
cpi_dict = {
	'1990': 1.980824032,
	'1991': 1.900345102,
	'1992': 1.844479154,
	'1993': 1.791597346,
	'1994': 1.746069601,
	'1995': 1.698421743,
	'1996': 1.650055254,
	'1997': 1.612363202,
	'1998':	1.587797546,
	'1999':	1.553487395,
	'2000':	1.50296748,
	'2001':	1.461383399,
	'2002':	1.438638132,
	'2003':	1.406581522,
	'2004':	1.370095289,
	'2005':	1.325197133,
	'2006':	1.283784722,
	'2007':	1.248232389,
	'2008':	1.202078002,
	'2009':	1.206369997,
	'2010':	1.18690153,
	'2011':	1.150583047,
	'2012':	1.127255068,
	'2013':	1.110981855,
	'2014':	1.09324733,
	'2015':	1.09195121,
	'2016':	1.078347715,
	'2017':	1.055854275,
	'2018':	1.030680148,
	'2019':	1.012336842,
	'2020':	1,
	'2021':	0.960376861
	}

### List of EAL column names
eal_cols = [
	'fld_eal_base_noFR_mid_fs_%s_m', # Use this as baseline
	'fld_eal_base_noFR_mid_fs_%s_l',
	'fld_eal_base_noFR_mid_fs_%s_h',
	'fld_eal_base_noFR_low_fs_%s_m',
	'fld_eal_base_noFR_low_fs_%s_l',
	'fld_eal_base_noFR_low_fs_%s_h',
	'fld_eal_base_noFR_high_fs_%s_m',
	'fld_eal_base_noFR_high_fs_%s_l',
	'fld_eal_base_noFR_high_fs_%s_h',
	'fld_eal_strat_all_mid_fs_%s_m',
	'fld_eal_strat_all_mid_fs_%s_l',
	'fld_eal_strat_all_mid_fs_%s_h',
	'fld_eal_strat_all_low_fs_%s_m',
	'fld_eal_strat_all_low_fs_%s_l',
	'fld_eal_strat_all_low_fs_%s_h',
	'fld_eal_strat_all_high_fs_%s_m',
	'fld_eal_strat_all_high_fs_%s_l',
	'fld_eal_strat_all_high_fs_%s_h',
	]