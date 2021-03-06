import re
import nltk
from nltk.corpus import stopwords
import math

class Document(object):
    '''
    Basic unit of document
    Attr:
        DID: int, document ID
        document_word_num: int, the total number in this document
        word_bag: dictionary, contains the words and a list of its position in document
        e.g. {'computer': [1,3]
              'milk': [7]}
    '''
    def __init__(self, DID, document_word_num, word_bag):
        self.DID = DID
        self.document_word_num = document_word_num
        self.word_bag = word_bag

    def show(self):
        print(self.word_bag)


class IndexTermPosting(object):
    '''
    Contains the df and its posting list of a single word index_term.
    Attr:
        total_num: int, the number of this word index_term in a single document
        idf: int, the idf value of this word.
        posting_dict: dictionary, records the document and its positons which contain the current word.
        tfidf_list: list, records the individual tf-idf value for each document about the current word.

    '''
    def __init__(self, total_num):
        self.total_num = total_num
        self.idf = 0
        self.posting_dict = {}
        self.tfidf_list = []

    def show_posting_dict(self):
        for k,v in self.posting_dict.items():
            print(str(k) + ' -> ' + str(v))


class QueryResult(object):
    '''
    Store the query retrieve result. One QueryResult store one document result.
    Attr:
        query: list, the query sentence which contains several query words
        DID: int, indicates the corresponding DID
        rank: int, indicates the rank of this result (range from 1 to 3)
        keyword_dict: dictionary, the top 5 most important keyword in this document.
        unique_keyword_num: int, the number of keyword which only appear in this document
        magnitude: float, the magnitude of document vector in vector space
        similarity_score: float: the cosine similarity score between document and query
    '''
    def __init__(self, query, DID):
        self.query = query
        self.rank = 0
        self.DID = DID
        self.keyword_dict = {}
        self.unique_keyword_num = 0
        self.magnitude = 0
        self.similarity_score = 0

    def show(self):
        print('For query: ')
        print(self.query)
        print('The top ' + str(self.rank + 1) + ' result is: ')
        print('DID: ' + str(self.DID))
        for k,v in self.keyword_dict.items():
            print(str(k) + ' -> ' + str(v))
        print('Number of unique keywords in document: ' + str(self.unique_keyword_num))
        print('Magnitude of the document vector: ' + str(self.magnitude))
        print('Similarity score: ' + str(self.similarity_score))
        print('----------')


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
    stop_words.append('said')
    stop_words.append('would')
    for sentence in raw_data:
        sentence = sentence.lower()
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
        total_document_num: int, the total word number inside the document
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
        document = Document(_index_in_line, len(line), word_bag)
        ret_document_list.append(document)
    total_document_num = len(ret_document_list)
    return ret_document_list, total_document_num


def generate_inverted_file(document_list, total_document_num):
    '''
    generate the inverted file according to the document list. 
    The structure of inverted_file is:
        {index_item1: IndexTermPosting1
         index_item2: IndexTermPosting2
         index_item3: IndexTermPosting3}
    Args:
        document_list: list, each line contains a Document instance 
        total_document_num: int, the total number of words in all documents.
    Returns: 
        inverted_file: dictionary, for each index_item, record its document frequency(df) and its posting list.
        len(inverted_file): int, the number of word in inverted file
    '''
    inverted_file = {}
    for document in document_list:
        for word in document.word_bag.items():
            if word[0] not in inverted_file:
                index_term_posting = IndexTermPosting(len(word[1]))
                inverted_file[word[0]] = index_term_posting
            else:
                inverted_file[word[0]].total_num += len(word[1])
            inverted_file[word[0]].posting_dict[document.DID] = word[1]
    for k, v in inverted_file.items():
        v.idf = math.log(total_document_num/(len(v.posting_dict)+1))
        for _index, document in enumerate(document_list):
            if _index in v.posting_dict:
                tf = len(v.posting_dict[_index]) / document.document_word_num
            else:
                tf = 0
            v.tfidf_list.append(tf * v.idf)
    return inverted_file, len(inverted_file)


def query_retrieve(processed_query, inverted_file):
    '''
    Do the retrieve function.
    Args:
        processed_query: list, each element contains a list of query words.
        inverted_file: list, {index_item1: {df1, posting_dict1}
                            index_item2: {df2, posting_dict2}
                            index_item3: {df3, posting_dict3}}
    Returns:
        ret_result: list, contains top three high similarity score results.
    '''
    ret_result = []
    for query in processed_query:
        candidate_list = []
        for term in query:
            for k,v in inverted_file[term].posting_dict.items():
                if k not in candidate_list:
                    candidate_list.append(k)
        document_weight_vector = compute_document_weight(inverted_file, candidate_list)
        query_result_list = compute_similarity_result(query, document_weight_vector, candidate_list)
        query_result_list = sorted(query_result_list, key=lambda x: x.similarity_score, reverse=True)
        for _index in range(3):
            query_result_list[_index].rank = _index
        ret_result.append(query_result_list[:3])
    return ret_result


def compute_document_weight(inverted_file, candidate_list):
    '''
    Compute the tfidf value for every candidates in every query. The weight of each query word is 
    represented by weight = (tf / tf_max) * log(N / df + 1)
    where tf_max indicates the number of word in the current document rather than the longest document.
    Args:
        inverted_file: list, used to compute df
        candidate_list: list, contains all the retrieved document ID in one query
    Returns:
        document_weight_vector: list, each element is a list of float, 
                                representing a document vector for this candidate
    '''
    document_weight_vector = []
    for candidate in candidate_list:
        single_weight_vector = []
        for word in inverted_file:
            single_weight_vector.append(inverted_file[word].tfidf_list[candidate])
        document_weight_vector.append(single_weight_vector)
    return document_weight_vector


def compute_similarity_result(query, document_weight_vector, candidate_list):
    '''
    Compute the cosine similarity between the query and document. 
    Compute all the results, create a QueryResult instance for each result, and store them in query_result_list.
    Args: 
        query: list, a list of query words
        document_weight_vector: list, each element is a list of float, 
                                representing a document vector for this candidate
        candidate_list: list, contains all the retrieved document ID in one query
    Returns:
        query_result_list: list, each element is an uncompleted QueryResult instance
    '''
    query_result_list = []
    query_vector = []
    for word in inverted_file:
        if word in query:
            query_vector.append(1)
        else:
            query_vector.append(0)

    for _index1, weight_vector in enumerate(document_weight_vector):
        d_times_q = 0
        for _index2 in range(len(query_vector)):
            d_times_q += query_vector[_index2] * weight_vector[_index2]
        mag_q = math.sqrt(sum(query_vector))
        mag_d = 0
        for num in weight_vector:
            mag_d += num ** 2
        mag_d = math.sqrt(mag_d)
        cosine_score = d_times_q /(mag_d * mag_q)
        query_result = QueryResult(query, candidate_list[_index1])
        query_result.magnitude = mag_d
        query_result.similarity_score = cosine_score
        query_result.keyword_dict = compute_top5_keyword(candidate_list[_index1], document_list, inverted_file)
        query_result.unique_keyword_num = compute_unique_num(candidate_list[_index1], document_list)
        query_result_list.append(query_result)
    return query_result_list


def compute_top5_keyword(DID, document_list, inverted_file):
    '''
    Compute the top 5 keyword in one document based on the tfidf value
    Args: 
        DID: int, document ID
        document_list: list, used to compute tf
        inverted_file: list, used to compute df
    Returns:
        ret_result: list, contains top 5 keyword and its posting list
    '''

    word_dict = {}
    ret_result = []
    document = document_list[DID]
    for k, v in document.word_bag.items():
        word_dict[k] = inverted_file[k].tfidf_list[DID]
    word_dict = dict(sorted(word_dict.items(),key=lambda x: x[1], reverse=True))
    word_dict = list(word_dict)[:5]
    for word in word_dict:
        ret_result.append((word, inverted_file[word].posting_dict))
    return dict(ret_result)


def compute_unique_num(DID, document_list):
    '''
    Find the number of unique word in one document
    Args: 
        DID: int, document ID
        document_list: list, used to find document
    Returns:
        unique_num: int, the number of unique word
    '''
    unique_num = 0
    document = document_list[DID]
    for word, word_num in document.word_bag.items():
        if len(word_num) == 1:
            unique_num += 1
    return unique_num


if __name__ == "__main__":

    collection_data = read_data('collection-100.txt')
    query_data = read_data('query-10.txt')

    processed_data, _ = preprocess(collection_data)
    processed_query, _ = preprocess(query_data)

    document_list, total_document_num = generate_word_bag(processed_data)
    inverted_file, total_word_num = generate_inverted_file(document_list, total_document_num)

    retrieve_results = query_retrieve(processed_query, inverted_file)
    for result in retrieve_results:
        for item in result:
            item.show()

