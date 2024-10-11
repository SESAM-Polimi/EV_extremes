#%%
import os
import pandas as pd

def extract_year_from_column(column_name):
    
    return int(column_name.split('_')[-1])

def read_data(file_path):
    
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    
def extract_mineral_data(base_dir):

    minerals = ['cobal', 'coppe', 'graph', 'lithi', 'manga', 'nicke']   
    all_data = {mineral: [] for mineral in minerals}  
   
    
    for year in range(2002, 2024):
        folder_path = os.path.join(base_dir, str(year))
        for mineral in minerals:
            for file_name in os.listdir(folder_path):
                if mineral in file_name:  
                    file_path = os.path.join(folder_path, file_name)
                                   
                    df = read_data(file_path)                        

                    prod_column = [col for col in df.columns if col.startswith('Prod_t_') or col.startswith('Prod_kt_')][0]
                                    
                    year_data = extract_year_from_column(prod_column)                                                        
                    
                    if 'Reserves' in df.columns:
                        reserves_column = 'Reserves'
                    elif 'Reserves_t' in df.columns:
                        reserves_column = 'Reserves_t'
                    elif 'Reserves_kt' in df.columns:
                        reserves_column = 'Reserves_kt'
                    else:
                        raise KeyError(f"No reserves for {mineral} in {file_name}")
                   
                    for index, row in df.iterrows():
                        country = row['Country']
                        reserves_value = row[reserves_column] if reserves_column else None
                        all_data[mineral].append((country, year_data, reserves_value))

    for mineral in minerals:
        df = pd.DataFrame(all_data[mineral], columns=['Country', 'Year', 'Reserves'])
        df['Reserves'] = pd.to_numeric(df['Reserves'], errors='coerce')
                
        df_final = df.pivot_table(index='Country', columns='Year', values='Reserves', fill_value=0)
        all_data[mineral] = df_final  

    return all_data

#%%

base_directory = "C:/Users/debor/Politecnico di Milano/DENG-SESAM - Documenti/c-Research/c-Publications on-going/2024_Al Khourdajie_ExtremesBEV/Data"

mineral_data = extract_mineral_data(base_directory)

output_file = 'mineral_reserves_by_country.xlsx'
with pd.ExcelWriter(output_file) as writer:
    for mineral, df in mineral_data.items():
        df.to_excel(writer, sheet_name=mineral.capitalize(), index=True)

print(mineral_data)



# %%
