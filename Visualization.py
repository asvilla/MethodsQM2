import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataf = pd.read_csv('data\\Normalized_similarity.csv')
dataf.rename(columns={'Similarity_to_Base_Pool': 'Similarity'}, inplace=True)
dataf['Year'] = pd.to_datetime(dataf['Year'], format='%Y')
# Exclude the "Liberal Democrat" party from the dataframe
df= dataf[dataf['Party'] != 'Liberal Democrat']

# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Similarity', hue='Party', data=df)
plt.title('Similarity Over Time by Party')
plt.xlabel('Year')
plt.ylabel('Similarity')
plt.legend(title='Party', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()
# Create a line plot
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Normalized_Similarity', hue='Party', data=df)
plt.title('Normalized Similarity Over Time by Party')
plt.xlabel('Year')
plt.ylabel('Normalized Similarity')
plt.legend(title='Party', loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()
# Create a FacetGrid with one plot per party
g = sns.FacetGrid(df, col='Party', col_wrap=3, height=5, sharey=False)
g.map(sns.lineplot, 'Year', 'Similarity')
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Year', 'Similarity')
plt.show()

# Create a FacetGrid with one plot per party
g = sns.FacetGrid(df, col='Party', col_wrap=3, height=5, sharey=False)
g.map(sns.lineplot, 'Year', 'Normalized_Similarity')
g.set_titles(col_template='{col_name}')
g.set_axis_labels('Year', 'Normalized Similarity')

plt.show()
