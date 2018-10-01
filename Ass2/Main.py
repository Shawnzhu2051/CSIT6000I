import numpy as np
import re
import nltk
from nltk.corpus import stopwords


class Document(object):
    '''
    Basic unit of document
    Attr:
        DID: int, document ID
        word_bag: dictionary, contains the words and a list of its position in document
        e.g. {'computer': [1,3]
              'milk': [7]}
    '''
    def __init__(self, DID, word_bag):
        self.DID = DID
        self.word_bag = word_bag

    def show(self):
        print(self.DID)
        print(self.word_bag)


class IndexTermPosting(object):
    '''
    Contains the df and its posting list of a single word index_term.
    Attr:
        df: int, the number of this word index_term in a single document
    '''
    def __init__(self, df):
        self.df = df
        self.posting_dict = {}


class QueryResult(object):
    '''
    Store the query retrieve result. One QueryResult store one document result.
    Attr:
        query: list, the query sentence which contains several query words
        rank: the rank of query result, range from 0 to 2.
    '''
    def __init__(self, query, rank):
        self.query = query
        self.rank = rank
        self.DID = 0
        self.keyword_list = []
        self.unique_keyword_num = []
        self.magnitude = 0
        self.similarity_score = []


def read_data(FILENAME):
    '''
    Read the collected document and query data from txt files.
    Args:
        FILENAME: str, name of input file
    Returns: 
        ret_data: list, each line contains one line in .txt file
    '''

    ret_data = []
    with open(FILENAME, 'r') as f:
        for line in f.readlines():
            if line != '\n':
                line = line.strip('\n')
                ret_data.append(line)
    return ret_data


def preprocess(raw_data):
    '''
    Preprocess the input raw data in following steps and compute the total word number.
    1, discard punctuation marks by regular expression(re).
    2, split the document line by spaces.
    3, remove the words that have less than 4 characters.
    4. remove ending “s” from a word
    5. remove stopwords by nltk.stopword
    Args:
        raw_data: list, each line contains one document or one query
    Returns: 
        ret_data: list, each line contains one processed document or query
        total_sum: int, total word number in whole data
    '''

    ret_data = []
    total_sum = 0
    # nltk.download('stopwords')  # if you use nltk.stopwords first time, you need to download it first
    stop_words = stopwords.words('english')
    for sentence in raw_data:
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
        total_sum += len(ret_sentence)
    return ret_data, total_sum


def generate_word_bag(processed_data):
    '''
    Generate document list from processed data. A document is a instance of class Document, 
    contains the DID and its word bag.
    Args:
        processed_data: list, each line contains one processed document
    Returns: 
        ret_document_list: list, each line contains a Document instance
    '''
    ret_document_list = []
    for _index_in_line, line in enumerate(processed_data):
        word_bag = {}
        for _index_in_word, word in enumerate(line):
            if word not in word_bag:
                word_bag[word] = []
                word_bag[word].append(_index_in_word)
            else:
                word_bag[word].append(_index_in_word)
        document = Document(_index_in_line, word_bag)
        ret_document_list.append(document)
    #for item in ret_document_list:
    #    print(item.DID)
    #    print(item.word_bag)
    return ret_document_list


def generate_inverted_file(document_list):
    '''
    generate the inverted file according to the document list. 
    The structure of inverted_file is:
        {index_item1: {df1, posting_dict1}
         index_item2: {df2, posting_dict2}
         index_item3: {df3, posting_dict3}}
    e.g. {'computer': {3, {0:[2,8], 5:[1]}}}
    Args:
        document_list: list, each line contains a Document instance 
    Returns: 
        inverted_file: dictionary, for each index_item, record its document frequency(df) and its posting list.
    '''
    inverted_file = {}
    for document in document_list:
        for word in document.word_bag.items():
            if word[0] not in inverted_file:
                index_term_posting = IndexTermPosting(len(word[1]))
                inverted_file[word[0]] = index_term_posting
            else:
                inverted_file[word[0]].df += len(word[1])
            inverted_file[word[0]].posting_dict[document.DID] = word[1]
    '''
    for item in inverted_file.items():
        print(item)
        print(item[1].df)
        print(item[1].posting_dict)
        print('----------')
    '''
    return inverted_file


def query_retrieve(processed_query, inverted_file, document_list, total_sum):
    '''
        Do the retrieve function.
        Args:
            processed_query: list, each element contains a list of query words.
            inverted_file: list, {index_item1: {df1, posting_dict1}
                                index_item2: {df2, posting_dict2}
                                index_item3: {df3, posting_dict3}}
            document_list: list, each element contains a Document instance 
            total_sum: the total number of all the words
        Returns: 
            
        '''
    for query in processed_query:
        candidate_list = []
        for term in query:
            for k,v in inverted_file[term].posting_dict.items():
                if k not in candidate_list:
                    candidate_list.append(k)
        document_weight_vector = compute_weight_by_tfidf(query,
                                                         inverted_file,
                                                         document_list,
                                                         candidate_list,
                                                         total_sum)
        cosine_similarity = compute_cosine_similarity(query, document_weight_vector)


def compute_weight_by_tfidf(query, inverted_file, document_list, candidate_list, total_sum):
    document_weight_vector = []
    for term in query:
        for candidate in candidate_list:
            if term in 
            idf = 0

    return document_weight_vector


def compute_cosine_similarity(query, document_weight_vector):
    query_weight = []
    for i in range(query):
        query_weight.append(1)



if __name__ == "__main__":
    collection_data = read_data('collection-100.txt')
    query_data = read_data('query-10.txt')

    processed_data, total_sum = preprocess(collection_data)
    processed_query, _ = preprocess(query_data)

    document_list = generate_word_bag(processed_data)

    inverted_file = generate_inverted_file(document_list)

    query_retrieve(processed_query, inverted_file, document_list, total_sum)
