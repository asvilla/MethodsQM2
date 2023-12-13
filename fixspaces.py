import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import os
import time
from sklearn.metrics.pairwise import cosine_similarity
import re

#this code is just to fix a format issue thath i had in the data with embeddings

#load data
current_directory = os.getcwd()
data_path = os.path.join(current_directory, 'data\\data_with_embeddings2.csv')

data = pd.read_csv(data_path)

def remove_space(embeddings_str):# Function to remove space after the opening bracket
    return embeddings_str.replace('[ ', '[')

data['Embeddings'] = data['Embeddings'].apply(remove_space)
data['Embeddings'] = data['Embeddings'].apply(lambda x:  re.sub(r"\s+", ", ", x))
data.to_csv('data\\data_embeddings.csv', index=False)