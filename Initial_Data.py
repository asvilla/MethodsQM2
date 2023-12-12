import pandas as pd
import pylab
import os
#import data into pandas dataframes
current_directory = os.getcwd()
bps_path = os.path.join(current_directory, "data\\speeches_bps.csv")
ukpol_path= os.path.join(current_directory,"data\\speeches_ukpol.csv")
partyinfo_path=os.path.join(current_directory,"data\\MPs_1970_complete.csv")
partyinfo_data= pd.read_csv(partyinfo_path,sep=';')
bps_data=  pd.read_csv(bps_path,index_col=0)
ukpol_data=  pd.read_csv(ukpol_path,index_col=0)
#PARTYINFO
#selecting relevant info from partyinfo dataset and removing polititians with no party info
#print(partyinfo_data.columns) this was to explore the columms in the dataframe
selected_columns = ['itemLabel','partyLabel']
partyinfo_newdf = partyinfo_data.loc[:, selected_columns]
partyinfo_newdf.columns = ['Name','Party']
# Identify rows with NaN values in 'name' column
partyinfo_newdf = partyinfo_newdf.dropna(subset=['Name'])
partyinfo_newdf['Name'] = partyinfo_newdf['Name'].str.lower()
# Check for duplicate names in partyinfo_newdf
partyinfo_newdf = partyinfo_newdf.sort_values(by='Name')
partyinfo_newdf = partyinfo_newdf.drop_duplicates(subset='Name', keep='first')

#UKPOL
#ukpol organizing and crossreferencing#
ukpol_data = ukpol_data.dropna(subset=['Year'])
ukpol_data['Year'] = ukpol_data['Year'].astype(int)
ukpol_data['Speaker'] = ukpol_data['Speaker'].str.lower()
#Meging ukpol and party
merged_df = pd.merge(ukpol_data, partyinfo_newdf, left_on='Speaker', right_on='Name', how='left')
merged_df = merged_df.drop('Name', axis=1)
merged_df= merged_df.dropna(subset=['Party'])
ukpol_subset = merged_df[['Speaker', 'Party', 'Speech','Year']].copy()
#bps data set organizing
bps_data['Speaker'] = bps_data['Speaker'].str.replace(r'\s*\([^)]*\)', '', regex=True) # just keeping the name in the speaker column
bps_data['Date'] = pd.to_datetime(bps_data['Date'], errors='coerce')
bps_data['Year'] = bps_data['Date'].dt.year
bps_subset = bps_data[['Speaker', 'Party', 'Speech','Year']].copy()
#join the datasets
result_df = pd.concat([bps_subset, ukpol_subset], ignore_index=True)
result_df = result_df.sort_values(by='Speaker')
result_df.to_csv("data\\SpeechesFullData.csv")
# Print the result DataFrame
print(result_df)