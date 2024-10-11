# IAM COMPACT - Out-of-ordinary extremes study
## Analysis on EV prices in case of disruptive increase in EV sales

This repository allows to perform an extensive analysis to estimate future EV prices based on projected EV sales.
Such projections are provided by other models within the IAM COMPACT consortium (e.g. GCAM) differentiated in 3 scenarios:
'High sales', 'Medium sales' and 'Low sales'

Link to the project website: https://www.iam-compact.eu

  
- - -
### Model use
The model can be used by installing the environment.yml file and running this notebook.
   

- - -
### Rationale of the model
The workflow of the model is here described:

- Global EV sales affect global minerals production. A linear regression model predict future prices of minerals based on a correlation fitted on historical mineral price and mineral production.

- Mineral prices would affect EVs batteries' prices. However, scientific literature argues such impact would be limited: for instance, doubling lithium price would mean 5% increase in battery prices, while a logaritmic price decrease is expected for batteries (Penisa et al. https://doi.org/10.3390/en13205276). The model currently implement such logaritmic trend to the historical battery prices. The future battery prices also account for the above-mentioned moderate impact brought by minerals prices. Differentr logaritmic trend could be explored by changing c0_avg and c1 parameters in the inputs/assumptions.yml file.

- Historical EVs prices are given. For the last historical year, the model calculates the fraction of the cost of the EV not attributable to the battery. Such cost would be kept constant in future years. This cost, summed to the future estimated battery prices multiplied by the specific battery capacity of each type of vehicle, would yield the future vehicle price
  
  
- - -  
### Model sets
The sets of the model are numerous and here outlined:
- model: by models, IAM COMPACT models providing vehicles future sales projections are intended here.
- scenario: each model provide vehicles sales projection according to different scenarios; a historical scenario is also provided to account for past timeseries (HIST).
- subsector: type of vehicles considered (e.g. Car, Mini Car, Light truck...); each model may differ in this set.
- region: all regions considered; each model may differ in this set.
- year: list of past and future years.
- market_share: additional scenario leverage allowing to simulate different future batteries chemistries market shares.
  
  
- - -  
### Inputs
The model inputs are stored in the "inputs" folder.
Three .yml files collect core information and assumptions:
- paths.yml: defines directories for input and results files
- new_names.yml: allows to rename any set provided by other model in a more readable label
- assumptions.yml: allows to provide core assumptions like occupancy rates of vehicles, vehicles battery capacities, minerals contents by chemistry
- data/vehicles folder: it includes data about vehicles production (e.g. sales projected by models + historical timeseries) and historical prices
- data/minerals folder: it contains an excel file for each mineral, including their historical production and prices. Reserves are currently not used
- data/batteries folder: the chemistry folder includes, for each chemistry, the historical price and capacity. The market_shares excel file allows to provide future estimations of time-evolving market shares of different chemistries
  
  
- - -    
### Results
The model exports results of prices and production for vehicles, batteries, and minerals in the results folder. Production represents "sales" for vehicles and "cumulative capacity" for batteries. The results are then imported into a PowerBi report to be visualized. The sets excel files allows for creating categories and aggregations
