# Unpriced climate risk and the potential consequences of overvaluation in US housing markets

This repository contains the code used to perform the analyses described in the following paper:

Gourevitch, J., Kousky, C., Liao, Y., Nolte, C., Pollack, A., Porter, J., and Weill, J. (2023) Unpriced climate risk and potential consequences of overvaluation in US housing markets. Nature Climate Change.

This repository is associated with the following DOI: 


## Abstract

Climate change impacts threaten the stability of the US housing market. In response to growing concerns that increasing costs of flooding are not fully captured in property values, we quantify the magnitude of unpriced flood risk in the housing market by comparing the empirical and economically efficient prices for properties at risk. We find that residential properties exposed to flood risk are overvalued by $121 – $237 billion, depending on the discount rate. In general, highly overvalued properties are concentrated in counties along the coast with no flood risk disclosure laws and where there is less concern about climate change. Low-income households are disproportionately at risk of losing home equity from price deflation, and municipalities that are heavily reliant on property taxes for revenue are vulnerable to budgetary shortfalls. The consequences of these financial risks will depend on policy choices that influence who bears the costs of climate change.


## Getting started

### Dependencies

* binsreg==1.0.0
* geopandas==0.10.1
* linearmodels==4.22
* matplotlib==3.3.3
* numpy==1.19.5
* pandas==1.3.5
* pyarrow==6.0.0
* scipy==1.6.0
* seaborn==0.11.2
* statsmodels==0.13.1
* twilio==7.3.2


### Project scripts

`main.py` contains a high-level function where the user can specify which functions to execute.

`driver.py` executes all user-specified functions.

`preprocessing.py` contains functions used to prepare data for analysis.

`stats.py` contains the functions used to conduct the statistical analyses.

`postprocessing.py` contains the functions used to analyze the output data.

`figures.py` contains the functions used to create the figures.

`paths.py` initializes paths to data inputs and outputs.

`utils.py` contains miscellaneous utility functions.

`references.py` contains miscellaneous reference data.


### How to use

`main.py` can be called by the user from the command line. `main.py` subsequently calls the 'execute' function in `driver.py`, which is responsible for executing all other functions.
