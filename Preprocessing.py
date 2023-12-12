import pandas as pd
import os
import spacy
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
import time

# Record the start time
start_time = time.time()

#import our joined data into pandas dataframe
current_directory = os.getcwd()
data_path = os.path.join(current_directory, "data\\filtered_dataset.csv")
data= pd.read_csv(data_path)

#First of all subsampleling#
combined_strata = data['Party'].astype(str) + '_' + data['Year'].astype(str)
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
for train_index, test_index in stratified_split.split(data, combined_strata):
    sampled_data = data.iloc[test_index]
#Preprocces speeches
nlp = spacy.load('en_core_web_sm') # Load the spaCy English model

def preprocess_text(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_.lower() for token in doc if not token.is_punct]  # Lemmatization and lowercase
    stop_words = spacy.lang.en.STOP_WORDS
    tokens_without_stopwords = [token for token in lemmatized_tokens if token not in stop_words and len(token) > 2]
    tokens_without_short_words = [token for token in tokens_without_stopwords if len(token) > 2]
    tokens_without_digits = [token for token in tokens_without_short_words if not token.isdigit()]  # Removing digits
    preprocessed_text = ' '.join(tokens_without_digits)
    print("::")
    return preprocessed_text


#Apply the peprocessing function to the "Speech" column of the subset
sampled_data['Processed_Speech'] = sampled_data['Speech'].apply(preprocess_text)

# Display the processed subset of data
print(sampled_data[['Speech', 'Processed_Speech']])
# Save the processed subset to a CSV file
sampled_data.to_csv('data\\processed_set.csv', index=False)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {elapsed_time:.2f} seconds")