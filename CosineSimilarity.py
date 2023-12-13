import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import re
import ast
# Record the start time
start_time = time.time()

#load data
current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'data\\data_embeddings.csv')

data = pd.read_csv(data_path)
data['Embeddings'] = data['Embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
print(data.columns)
# Create a new DataFrame to store similarity scores
result_data = pd.DataFrame()

## similarity part
target_party = 'Liberal Democrat'

for year in range(2005, 2020):
    print(year)
    # Step 1: Create a Base Pool for the Specific Year and Party
    base_pool = data[(data['Year'] == year) & (data['Party'] == target_party)]
    base_embeddings = np.vstack(base_pool['Embeddings'].to_numpy())

    # Step 2: Compare Individual Speeches to the Base Pool
    def compare_to_base_pool(embeddings, base_embeddings, ID):
        similarities = cosine_similarity(embeddings.reshape(1, -1), base_embeddings).flatten()
        print(ID)
        print(year)
        return np.mean(similarities)

    current_year_data = data[data['Year'] == year]  # Filter data for the current year
    # Apply the comparison function to each row
    current_year_data['Similarity_to_Base_Pool'] = current_year_data.apply(
        lambda row: compare_to_base_pool(row['Embeddings'], base_embeddings,row["Speech_ID"]), axis=1)
    # Append the current year data to the result data
    result_data = result_data.append(current_year_data, ignore_index=True)
    
selected_columns = ['Speech_ID', 'Speaker', 'Party', 'Constituency', 'Year','Similarity_to_Base_Pool']
filtered_dataset = result_data[selected_columns]
result_data.to_csv('data\\output_data_with_similarity.csv', index=False)
filtered_dataset.to_csv('data\\similarity_data.csv', index=False)
###
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")