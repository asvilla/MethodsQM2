import pandas as pd

df = pd.read_csv('data\\similarity_data.csv')
# Find the maximum similarity value for the "Liberal Democrat" party for each year
max_similarity_ld = df[df['Party'] == 'Liberal Democrat'].groupby('Year')['Similarity_to_Base_Pool'].max()
# Normalize similarity values for each party relative to "Liberal Democrat"
df['Normalized_Similarity'] = df.apply(lambda row: row['Similarity_to_Base_Pool'] / max_similarity_ld[row['Year']], axis=1)
df.to_csv('data\\Normalized_similarity.csv', index=False)