import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#import our joined data into pandas dataframe
current_directory = os.getcwd()
speech_data_path = os.path.join(current_directory, "data\\data_embeddings.csv")

speech_data= pd.read_csv(speech_data_path)
speech_data['Party'] = speech_data['Party'].str.lower() # Convert all values in the 'Party' column to lowercase
speech_data['Party'] = speech_data['Party'].str.replace(' party', '')# Remove the word 'party' from all values in the 'Party' column
speech_data['Party'] = speech_data['Party'].astype('category')
num_rows = speech_data.shape[0]
print("Number of rows:", num_rows)
print(speech_data)
# Filter data for the years since 2000
speech_data_since_2000 = speech_data[speech_data['Year']>= 2000]

# Plot: Number of Different Speakers per Year
speakers_per_year = speech_data_since_2000.groupby('Year')['Speaker'].nunique()
speakers_per_year.plot(kind='bar', title='Number of Different Speakers per Year')
plt.xlabel('Year')
plt.ylabel('Number of Speakers')
plt.show()

# Plot: Number of Speeches per Year
speeches_per_year = speech_data_since_2000.groupby('Year')['Speech_ID'].count()
speeches_per_year.plot(kind='bar', title='Number of Speeches per Year')
plt.xlabel('Year')
plt.ylabel('Number of Speeches')
plt.show()
# Plot: Number of Speeches per Party per Year
speeches_per_party_per_year = speech_data_since_2000.groupby(['Year', 'Party'])['Speech_ID'].count().unstack()
speeches_per_party_per_year.plot(kind='bar', stacked=True, title='Number of Speeches per Party per Year')
plt.xlabel('Year')
plt.ylabel('Number of Speeches')
plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# Filter data for the party 'liberal democrat'
liberal_democrat_data = speech_data_since_2000[speech_data_since_2000['Party'] == 'liberal democrat']

# Group by year and count the number of speeches
speeches_per_year_libdem = liberal_democrat_data.groupby('Year')['Speech_ID'].count()

# Plot: Speeches per Year for 'liberal democrat'
plt.figure(figsize=(10, 6))
speeches_per_year_libdem.plot(kind='bar', color='skyblue')
plt.title('Speeches per Year for Liberal Democrats')
plt.xlabel('Year')
plt.ylabel('Number of Speeches')
plt.show()