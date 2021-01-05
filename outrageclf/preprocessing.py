import numpy as np
from . import helpers
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from joblib import dump, load
import urllib

glove_default_url = 'https://worksheets.codalab.org/rest/bundles/'\
                     '0x4090ba96b8a444c2a44b2c47884c25f2/'\
                     'contents/blob/glove.twitter.27B.50d.txt'
'''
Get lemmatize hashtag
    - Create a lemmatized + hashtag features for text vector
    - Input: a text vector
    - Output: a vector of lemmatized keywords + hashtags (if exist)
'''

def get_lemmatize_hashtag(text_vector):
    hashtag_ls = [helpers.get_hashtag(text) for text in text_vector]
    wn_lemmatize_ls = [helpers.tweet_process(text) for text in text_vector]
    hashtag_lemmatize = [' '.join([x for x in lemma.split(" ") + hashtag.split(" ")]) 
    for lemma, hashtag in zip(wn_lemmatize_ls, hashtag_ls)]
    return hashtag_lemmatize



'''
Word embedding object:
    * MAXLEN defaults to 50, we currently don't support customization

    * TRAINING:
        - Users' responsibility to submit a valid tokenizer path with .joblib format

    * USING A PRETRAINED WORD EMBEDDING:
        - Please contact the Crockett lab for access to the tokenizer
        - Users' responsbility to submit a valid tokenizer path with .joblib format
'''

class WordEmbed:
    def __init__(self):
        self.tokenizer_path = None
        self.tokenizer = None

    def _get_pretrained_tokenizer(self, path):
        self.tokenizer_path = path
        self.tokenizer = load(self.tokenizer_path)
        print ("Loaded pre-trained tokenizer at:", path)

    def _train_new_tokenizer(self, text_vector, saving_path):
        self.tokenizer_path = saving_path
        embedding_tokenizer = Tokenizer()
        embedding_tokenizer.fit_on_texts(text_vector)

        
        self.tokenizer = embedding_tokenizer
        dump(embedding_tokenizer, self.tokenizer_path)
        print ("Trained and saved new tokenizer at:",
            self.tokenizer_path)

    def _get_embedded_vector(self, text_vector):
        embedded = pad_sequences(self.tokenizer.texts_to_sequences(text_vector),
            padding='post',
            maxlen=50)
        return embedded



'''
Create Embedding matrix
- based on pre-defined tokenizer
- currently only supported Glove 50d Twitter
- will be updated to support different embedding in the future

Input:  - word_index: from an associated Tokenizer, called from preprocessing.py
        - filepath: file path to a pretrained embedding e.g Glove 50d Twitter

Result: - An embedding matrix for embedding based model such as LSTM, GRU
        * It is strictly associated with the Tokenizer used in the word_index argument
        * User's responsibility to make sure they are correct
'''

def create_embedding_matrix(word_index, filepath):
    embedding_dim = 50
    # Adding again 1 because of reserved 0 index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix
    


'''
Create Embedding matrix from default url link

Similar to create_embedding_matrix
but use a online storage of Glove 27B 50d embedding
'''

def create_embedding_matrix_default(word_index):
    embedding_dim = 50
    # Adding again 1 because of reserved 0 index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    file = urllib.request.urlopen(glove_default_url)

    for line in file:
        word, *vector = line.split()
        if word.decode() in word_index:
            idx = word_index[word.decode()] 
            embedding_matrix[idx] = np.array(
                vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix
