import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import time
from sklearn.metrics.pairwise import cosine_similarity

# Record the start time
start_time = time.time()

current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'data\\processed_set.csv')
data = pd.read_csv(data_path)

# Convert the 'Embeddings' column back to NumPy arrays
data['Embeddings'] = data['Embeddings'].apply(lambda x: np.array(eval(x)))

## similarity part
target_party = 'Liberal Democrat'

for year in range(2005, 2020):
    # Step 1: Create a Base Pool for the Specific Year and Party
    base_pool = data[(data['Year'] == year) & (data['Party'] == target_party)]
    base_embeddings = np.vstack(base_pool['Embeddings'].to_numpy())

    # Step 2: Compare Individual Speeches to the Base Pool
    def compare_to_base_pool(embeddings, base_embeddings):
        similarities = cosine_similarity(embeddings.reshape(1, -1), base_embeddings).flatten()
        return similarities

    current_year_data = data[data['Year'] == year] # Filter data for the current year

    # Apply the comparison function to each row
    current_year_data['Similarity_to_Base_Pool'] = current_year_data.apply(
        lambda row: compare_to_base_pool(row['Embeddings'], base_embeddings), axis=1)
    # Save the similarity scores in the DataFrame
    data.loc[data['Year'] == year, 'Similarity_to_Base_Pool'] = current_year_data['Similarity_to_Base_Pool']
# Save the updated DataFrame to a CSV file
data.to_csv('output_data_with_similarity.csv', index=False)
###
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")