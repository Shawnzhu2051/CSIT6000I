import numpy as np
import re
import nltk
from nltk.corpus import stopwords

def read_data(FILENAME):
    collection_data = []
    with open(FILENAME, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                collection_data.append(line)
    return collection_data

def preprocess(collection_data):
    ret_data = []
    # nltk.download('stopwords')  # if you use nltk.stopwords first time, you need to download it first
    stop_words = stopwords.words('english')
    for sentence in collection_data:
        punctuation = '[,.!\'\"/+()-]'
        updated_sentence = re.sub(punctuation, ' ', sentence)
        updated_sentence = updated_sentence.split()
        ret_sentence = []
        for word in updated_sentence:
            if len(word) >= 4 and word not in stop_words:
                if word[-1] != 's':
                    ret_sentence.append(word)
                else:
                    ret_sentence.append(word[:-1])
        ret_data.append(ret_sentence)
    return ret_data


if __name__ == "__main__":
    collection_data = read_data('collection-100.txt')
    processed_data = preprocess(collection_data)