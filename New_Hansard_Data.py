import pandas as pd
import os
#import our joined data into pandas dataframe
current_directory = os.getcwd()
speech_data_path = os.path.join(current_directory, "data\\hansard-speeches-v310.csv")
speech_data= pd.read_csv(speech_data_path)
# Filter data for the years since 2005 until 2019
# Select only the desired columns
selected_columns = ['speech', 'display_as', 'party', 'constituency', 'year']
filtered_dataset = speech_data[selected_columns]
# Rename the columns
filtered_dataset = filtered_dataset.rename(columns={
    'speech': 'Speech',
    'display_as': 'Speaker',
    'party': 'Party',
    'constituency': 'Constituency',
    'year': 'Year'
})


num_rows = filtered_dataset.shape[0]
print("Number of rows:", num_rows)
# Filter data 
filtered_dataset = filtered_dataset.dropna(subset=['Year'])
filtered_dataset = filtered_dataset[(filtered_dataset['Year'] >= 2005) & (filtered_dataset['Year'] <= 2019)]
filtered_dataset = filtered_dataset.dropna(subset=['Speaker'])
filtered_dataset = filtered_dataset.dropna(subset=['Speech'])
filtered_dataset = filtered_dataset.dropna(subset=['Party'])
filtered_dataset = filtered_dataset[filtered_dataset['Party'] != 'Speaker'] # Remove rows where the value in the 'Party' column is 'Speaker'
party_counts = filtered_dataset['Party'].value_counts()
top_parties = party_counts.head(7).index.tolist()
filtered_dataset = filtered_dataset[filtered_dataset['Party'].isin(top_parties)]
filtered_dataset.to_csv('data\\filtered_dataset.csv', index=False)
print(filtered_dataset)
num_rows = filtered_dataset.shape[0]
print("Number of rows:", num_rows)