import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

start_time = time.time()# Record the start time
#LOAD te procesed data and check for nulls and type
current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'data\\processed_set.csv')

data = pd.read_csv(data_path)

data = data.dropna(subset=['Processed_Speech'])
data['Processed_Speech'] = data['Processed_Speech'].astype(str)

##
#subset_size = 1000
# Create a random subset of the data with 1000 rows
#random_subset = data.sample(n=subset_size, random_state=42) 
random_subset = data
# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
# Function to embed text using BERT
def embed_text(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = bert_model(**tokens)
    embeddings = output['last_hidden_state'].mean(dim=1).squeeze().numpy()
    print(".")
    return embeddings

#apply to dataset
random_subset['Embeddings'] = random_subset['Processed_Speech'].apply(embed_text)
selected_columns = ['Speech_ID','Speaker', 'Party', 'Constituency', 'Year', 'Embeddings']
random_subset[selected_columns].to_csv('data\\data_with_embeddings2.csv', index=False)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
