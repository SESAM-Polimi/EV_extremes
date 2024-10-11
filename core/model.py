#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math 

years = range(1990,2101)
_acceptables_units = ['million ton-km','million pass-km']

#%%
class EVs:

    def __init__(
            self,
            assumptions: dict,
    ):
        
        """
        Initialize the class
        Args:      
            assumptions: dict: dictionary with the assumptions for the conversion of km to vehicles
        """

        self.production = pd.DataFrame()  
        self.prices = pd.DataFrame()

        self.assumptions = assumptions
        self.unit_conversion_factors = {}


    def parse_production(
            self,
            path: str,
            model:str = None,
    ):
        df = pd.read_excel(path)

        # Keep just new sales and removing residual yearly vehicles fleet
        other_columns = [c for c in df.columns if c not in years]
        df.set_index(other_columns, inplace=True)

        # Reshape data
        df.columns.names = ['year']
        df = df.stack()
        df = df.to_frame()
        df.columns = ['value']
        df.reset_index(inplace=True)

        # Get annual sales        
        if model == 'GCAM':   
            df['value'] /= 5

        # Rename and drop columns
        df = df[~df['technology'].str.contains('BEV_state_target,year|BEV_national_target,year')]

        df['technology'] = [int(t.split('=')[-1]) for t in df['technology']]        
        
        df = df[df['technology'] == df['year']]            
    
        df = df.drop(columns=['technology','sector'])


        if model != None:
            df['model'] = model
        else:
            df['model'] = 'not specified'

        self.production = pd.concat([self.production,df], axis=0)


    def rename_column_values(
            self,
            new_names: dict,
    ):
        """
        Rename the values of the columns in the dataframe

        Args:
            new_names (dict): Dictionary with the new names
        """

        for column in self.production.columns:
            if column in new_names.keys():
                self.production[column] = self.production[column].map(new_names[column])


    def parse_historical_prices(
            self,
            path:str,
    ):
        
        df = pd.read_excel(path)
        df['scenario'] = 'HIST'
        df['model'] = 'HIST'
        df['market_share'] = 'HIST'

        self.prices = pd.concat([self.prices,df], axis=0)


    def convert_km_to_vehicles(
            self,
            units_to_convert: list,
            unit_col: str = 'Units',
            new_unit: str = 'million vehicles',
    ):
        """
        Convert the units of the values in the dataframe

        Args:
            unit_to_convert (list): List with the units to convert
            unit_col (str): Name of the column with the units. Default is 'Units'
        """

        if any(unit not in _acceptables_units for unit in units_to_convert):
            raise ValueError(f'Some units are not acceptable. Acceptable units are {_acceptables_units}')

        for unit in units_to_convert:            
            if unit not in self.unit_conversion_factors.keys():
                if unit == 'million pass-km':
                    self.km_to_vehicles(
                        unit_to_convert = unit,
                        yearly_mileage = self.assumptions['passenger_yearly_mileage'],
                        occupancy_rate = self.assumptions['passenger_occupancy_rate'],
                    )
                elif unit == 'million ton-km':
                    self.km_to_vehicles(
                        unit_to_convert = unit,
                        yearly_mileage = self.assumptions['freight_yearly_mileage'],
                        occupancy_rate = self.assumptions['freight_occupancy_rate'],
                    )

            factor = self.unit_conversion_factors[unit]

            self.production['value'] = self.production.apply(
                lambda row: row['value'] * factor if row[unit_col] == unit else row['value'],
                axis=1
            )

        self.production[unit_col] = new_unit


    def km_to_vehicles(
            self,
            unit_to_convert: str,    
            yearly_mileage: float,
            occupancy_rate: float,
    )-> float:
        """
        Convert km to vehicles

        Args:
            unit_to_convert (str): Unit to be converted. Acceptables are 'million ton-km' and 'million passenger-km'
            yearly_mileage (float): Assumed yearly mileage per vehicle
            occupancy_rate (float): Assumed occupancy rate of the vehicle
        """

        self.unit_conversion_factors[unit_to_convert] = 1/(yearly_mileage*occupancy_rate)
    

    def set_last_hist_year(
            self,
            last_hist_year: int,
    ):
        data = self.production.copy()

        if any(data.query("year==@last_hist_year").model.unique()) != 'HIST':
            
            rows = list(data.query("year==@last_hist_year").index)

            new_data = data.query("year==@last_hist_year").copy()
            new_data['model'] = 'HIST'
            new_data['scenario'] = 'HIST'

            new_data.set_index([c for c in new_data.columns if c not in ['value']], inplace=True)
            new_data = new_data.groupby(list(new_data.index.names)).mean()

            new_data.reset_index(inplace=True)

            data = data.drop(rows)
            data = pd.concat([data,new_data], axis=0)

        self.production = data
        self.production = self.production.sort_values(by=['year'])
            
    def get_global_production(
            self,
            region: str='Global',
            excluded_cols: list = ['subsector'],
    ):
        
        data = self.production.query("region==@region")
        data.set_index([c for c in data.columns if c not in ['value']], inplace=True)

        to_group_cols = [c for c in data.index.names if c not in excluded_cols]

        data = data.groupby(to_group_cols).sum()
        data.reset_index(inplace=True)

        return data


    def estimate_future_prices(
            self,
            batteries,
            last_hist_year:int = 2019,
    ):
        
        regions = self.production.region.unique()
        models = self.production.model.unique()
        scenarios = self.production.scenario.unique()
        market_shares = batteries.prices.market_share.unique()
        years = [i for i in self.production.year.unique() if i > last_hist_year]
        subsectors = self.production.subsector.unique()
        chemistries = batteries.prices.chemistry.unique()

        future_prices = pd.DataFrame()

        for subsector in subsectors:
            vehicle_battery_capacity = batteries.assumptions['capacity'][subsector]

            for region in regions:
                vehicle_hist_price = self.prices.query("region==@region & subsector==@subsector & year==@last_hist_year").value.values[0]

                avg_vehicle_hist_battery_cost = 0
                avg_vehicles_non_battery_cost = 0

                for chemistry in chemistries:
                    battery_hist_price = batteries.prices.query("chemistry==@chemistry & scenario=='HIST' & model=='HIST' & market_share=='HIST' & year==@last_hist_year").value.values[0]
                    vehicle_hist_battery_cost = battery_hist_price * vehicle_battery_capacity
                    vehicles_non_battery_cost = vehicle_hist_price - vehicle_hist_battery_cost
                    market_share = batteries.market_shares.query("year==@last_hist_year and chemistry==@chemistry").value.values[0]

                    avg_vehicle_hist_battery_cost += vehicle_hist_battery_cost*market_share
                    avg_vehicles_non_battery_cost += vehicles_non_battery_cost*market_share
                

                for ms in market_shares:                        
                    for scenario in scenarios:                    
                        for model in models:
                            for year in years:
                                try:
                                    avg_vehicle_future_battery_cost = 0
                                    for chemistry in chemistries:
                                        market_share = batteries.market_shares.query("year==@year and chemistry==@chemistry and market_share==@ms").value.values[0]
                                        avg_vehicle_future_battery_cost += batteries.prices.query("chemistry==@chemistry & scenario==@scenario & model==@model & year==@year & market_share==@ms").value.values[0] * vehicle_battery_capacity * market_share

                                    avg_vehicle_future_price = avg_vehicle_future_battery_cost + avg_vehicles_non_battery_cost

                                    df = pd.DataFrame({
                                        'year': [year],
                                        'value': [avg_vehicle_future_price],
                                        'region': [region],
                                        'subsector': [subsector],
                                        'scenario': [scenario],
                                        'market_share': [ms],
                                        'model': [model],
                                        'Units': ['USD'],
                                    })

                                    future_prices = pd.concat([future_prices,df], axis=0)
                                except:
                                    pass
        
        self.prices = pd.concat([self.prices,future_prices], axis=0)


#%%
class Minerals:

    def __init__(
            self,
            assumptions: dict,
    ):
        
        self.prices = pd.DataFrame()       
        self.production = pd.DataFrame()
        self.reserves = pd.DataFrame()

        self.assumptions = assumptions    

    def parse_historical_price(
            self,
            path: str,
            mineral: str,
    ):
        
        df = pd.read_excel(path, sheet_name='Price')
        df['mineral'] = mineral
        df['scenario'] = 'HIST'

        self.prices = pd.concat([self.prices,df], axis=0)

    def parse_historical_production(
            self,
            path: str,
            mineral: str,
    ):
        
        df = pd.read_excel(path, sheet_name='Production')
        df['mineral'] = mineral
        df['scenario'] = 'HIST'
        
        self.production = pd.concat([self.production,df], axis=0)

    def get_production_for_ev(
            self,
            mineral: str,
            vehicle_production: pd.DataFrame,
    ):
        df = vehicle_production.copy()
        df['value'] *= self.assumptions[mineral]['content']
        df['mineral'] = mineral
        df['Units'] = 'ton'

        self.production_for_ev = df

    def get_future_production(
            self,
            mineral: str,
            last_valid_years: int,
            last_hist_year:int = 2019,
    ):
        hist_production = self.production.query("year<=@last_hist_year and mineral==@mineral")    
        
        hist_production_for_evs = self.production_for_ev.query("year<=@last_hist_year and mineral==@mineral and region=='Global'")
        hist_production_for_evs.set_index([c for c in hist_production_for_evs.columns if c not in ['value']], inplace=True)
        hist_production_for_evs = hist_production_for_evs.groupby([c for c in hist_production_for_evs.index.names if c!='subsector']).sum()
        hist_production_for_evs.reset_index(inplace=True)

        future_production = pd.DataFrame()

        avg_last_valid_years = hist_production['value'] - hist_production_for_evs['value']
        avg_last_valid_years = avg_last_valid_years[-last_valid_years:].mean()

        for scenario in self.production_for_ev.scenario.unique():
            
            if scenario != 'HIST':
                df_values = self.production_for_ev.query("mineral==@mineral and scenario==@scenario and region=='Global'").copy()
                df_values.set_index([c for c in df_values.columns if c not in ['value']], inplace=True)
                df_values = df_values.groupby([c for c in df_values.index.names if c!='subsector']).sum()
                df_values.reset_index(inplace=True)
                df_values = df_values['value'].values + avg_last_valid_years

                df_years = [y for y in self.production_for_ev.year.unique() if y> last_hist_year]
                df_scenario = [scenario for y in self.production_for_ev.year.unique() if y> last_hist_year]
                df_mineral = [mineral for y in self.production_for_ev.year.unique() if y> last_hist_year]
                df_units = ['ton' for y in self.production_for_ev.year.unique() if y> last_hist_year]

                df = pd.DataFrame({
                    'year': df_years,
                    'value': df_values,
                    'mineral': df_mineral,
                    'scenario': df_scenario,
                    'Units': df_units,
                })

                future_production = pd.concat([future_production,df], axis=0)
        
        self.production = pd.concat([self.production,future_production], axis=0)


    def estimate_future_prices(
            self,
            mineral: str,
            last_hist_year:int = 2019,
    ):
        
        X = self.production.query("mineral==@mineral and year<=@last_hist_year")['value'].values.reshape((-1, 1))
        Y = self.prices.query("mineral==@mineral and year<=@last_hist_year")['value'].values

        model = LinearRegression()
        model.fit(X, Y)

        scenarios = self.production.scenario.unique()
        for scenario in scenarios:
            if scenario != 'HIST': 
                X_future = self.production.query("mineral==@mineral and year>@last_hist_year and scenario==@scenario")['value'].values.reshape((-1, 1))
                Y_future = model.predict(X_future)

                df = pd.DataFrame({
                    'year': self.production.query("mineral==@mineral and year>@last_hist_year and scenario==@scenario")['year'].values,
                    'value': Y_future,
                    'mineral': mineral,
                    'scenario': scenario,
                    'Units': self.prices.query("mineral==@mineral and year<=@last_hist_year")['Units'].values[0],
                })

                self.prices = pd.concat([self.prices,df], axis=0)

# %%
class Batteries():
    
    def __init__(
            self,
            assumptions: dict,
    ):
        
        self.prices = pd.DataFrame()  
        self.production = pd.DataFrame()     

        self.assumptions = assumptions
    
    def parse_historical_price(
            self,
            path: str,
            chemistry: str,
    ):
        
        df = pd.read_excel(path, sheet_name='Price')
        df['chemistry'] = chemistry
        df['scenario'] = 'HIST'
        df['model'] = 'HIST'
        df['market_share'] = 'HIST'

        self.prices = pd.concat([self.prices,df], axis=0)

    def parse_historical_production(
            self,
            path: str,
            chemistry: str,
    ):
        
        df = pd.read_excel(path, sheet_name='Capacity')
        df['chemistry'] = chemistry
        df['scenario'] = 'HIST'
        df['model'] = 'HIST'
        
        self.production = pd.concat([self.production,df], axis=0)

    def parse_market_shares(
            self,
            path: str,
    ):
            
        projections = pd.read_excel(path, sheet_name=None)
        del projections['References']

        self.scenarios = list(projections.keys())
        self.market_shares = pd.DataFrame()

        for scenario,projection in projections.items():
            projection.set_index(['year','Units'],inplace=True)
            projection.columns.name = 'chemistry'
            projection = projection.stack()
            projection = projection.to_frame()
            projection.columns = ['value']  
            projection.reset_index(inplace=True)
            projection['market_share'] = scenario
            self.market_shares = pd.concat([self.market_shares,projection], axis=0)

    def get_future_production(
            self,
            chemistry: str,
            ev_sales: pd.DataFrame,
            last_hist_year:int = 2019,
    ):      
        
        future_ev_sales = ev_sales.query("year>=@last_hist_year")
        last_hist_year_ev = ev_sales.query("scenario=='HIST'").year.unique()[-1]

        future_capacity = pd.DataFrame()    
        
        merged_scenarios = []
        for batt_scenario in self.scenarios:
            for ev_sales_scenario in future_ev_sales.scenario.unique():
                if batt_scenario != 'HIST':
                    if ev_sales_scenario != 'HIST':
                        merged_scenarios += [f'{ev_sales_scenario}_{batt_scenario}']
        
        future_years = future_ev_sales.year.unique()
        subsectors = future_ev_sales.subsector.unique()

        for scenario in merged_scenarios:
            ev_sales_scenario = scenario.split('_')[0]
            batt_scenario = scenario.split('_')[1]
            for model in future_ev_sales.model.unique():
                if model != 'HIST':
                    for year in future_years:
                        if year == future_years[0]:
                            cap = self.production.query("year==@last_hist_year and chemistry==@chemistry").value.values[0]
                        else:
                            past_year_cap = future_capacity.query("year==@past_year and scenario==@ev_sales_scenario and market_share==@batt_scenario and chemistry==@chemistry").value.values[0]
                            cap = 0
                            market_share = self.market_shares.query("year==@year and market_share==@batt_scenario and chemistry==@chemistry").value.values[0]

                            for subsector in subsectors:
                                specific_cap = self.assumptions['capacity'][subsector]

                                if year <= last_hist_year_ev:
                                    if subsector not in ev_sales.query("year==@year and scenario=='HIST' and model=='HIST'")['subsector'].unique():
                                        ev_sold = 0
                                    else:
                                        ev_sold = ev_sales.query("year==@year and scenario=='HIST' and model=='HIST' and subsector==@subsector").value.values[0]
                                else:
                                    if subsector not in ev_sales.query("year==@year and scenario==@ev_sales_scenario and model==@model")['subsector'].unique():
                                        ev_sold = 0
                                    else:
                                        ev_sold = ev_sales.query("year==@year and scenario==@ev_sales_scenario and model==@model and subsector==@subsector").value.values[0]*5
                                
                                cap += specific_cap*market_share*ev_sold
                            
                            cap += past_year_cap

                        df = pd.DataFrame({
                            'year': [year],
                            'value': [cap],
                            'chemistry': [chemistry],
                            'scenario': [ev_sales_scenario],
                            'market_share': [batt_scenario],
                            'Units': ['GWh'],
                            'model': [model],
                        })

                        future_capacity = pd.concat([future_capacity,df], axis=0)
                        past_year = year.copy()
        
        self.production = pd.concat([self.production,future_capacity], axis=0)


    def estimate_future_prices(
            self,
            chemistry: str,
            minerals: Minerals,
            last_hist_year:int = 2019,
            mode:str = 'log',
    ):      

        future_capacity = self.production.query("chemistry==@chemistry and year>@last_hist_year")
        models = future_capacity.model.unique()
        scenarios = []
        for a in future_capacity.scenario.unique():
            for b in future_capacity['market_share'].unique():
                scenarios += [f"{a}_{b}"]
        years = future_capacity.year.unique()

        future_prices = pd.DataFrame()

        minerals_prices = minerals.prices

        if mode == 'log':  # source Penisa et al. https://doi.org/10.3390/en13205276            

            for model in models:  
                for scenario in scenarios:
                    mineral_scenario = scenario.split('_')[0]
                    batt_scenario = scenario.split('_')[1]
                    for year in years:
                        c0 = self.assumptions['c0_avg']
                        c1 = self.assumptions['c1']
                        cap = future_capacity.query("year==@year and model==@model and scenario==@mineral_scenario and market_share==@batt_scenario and chemistry==@chemistry").value.values[0]
                        year_pos = years.tolist().index(year)
                        try:
                            price = 10 ** (c0 + c1*math.log10(cap))
                            for mineral in minerals_prices.mineral.unique():
                                if year == years[0]:
                                    mineral_price_current_year = minerals_prices.query("mineral==@mineral and year==@year and scenario=='HIST'").value.values[0]
                                    mineral_price_previous_year = minerals_prices.query("mineral==@mineral and year==@last_hist_year and scenario=='HIST'").value.values[0]
                                elif year == years[1]:
                                    previous_year = years[0]
                                    mineral_price_current_year = minerals_prices.query("mineral==@mineral and year==@year and scenario==@mineral_scenario").value.values[0]
                                    mineral_price_previous_year = minerals_prices.query("mineral==@mineral and year==@previous_year and scenario=='HIST'").value.values[0]
                                else:
                                    previous_year = years[year_pos-1]
                                    mineral_price_current_year = minerals_prices.query("mineral==@mineral and year==@year and scenario==@mineral_scenario").value.values[0]
                                    mineral_price_previous_year = minerals_prices.query("mineral==@mineral and year==@previous_year and scenario==@mineral_scenario").value.values[0]

                                delta_mineral_price = (mineral_price_current_year - mineral_price_previous_year)/mineral_price_previous_year
                                delta_battery_price = delta_mineral_price * minerals.assumptions[mineral]['impact_on_battery_price']

                                price *= (1+delta_battery_price)

                        except:
                            price = 0

                        df = pd.DataFrame({
                            'year': [year],
                            'value': [price],
                            'chemistry': [chemistry],
                            'scenario': [mineral_scenario],
                            'market_share': [batt_scenario],
                            'Units': ['USD/kWh'],
                            'model': [model],
                        })

                        future_prices = pd.concat([future_prices,df], axis=0)

        self.prices = pd.concat([self.prices,future_prices], axis=0)




