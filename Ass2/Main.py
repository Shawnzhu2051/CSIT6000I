import re
import nltk
from nltk.corpus import stopwords
import math

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

    def show_df(self):
        print('df: ' + str(self.df))

    def show_posting_dict(self):
        for k,v in self.posting_dict.items():
            print(str(k) + ' -> ' + str(v))


class QueryResult(object):
    '''
    Store the query retrieve result. One QueryResult store one document result.
    Attr:
        query: list, the query sentence which contains several query words
        DID: int, indicate the corresponding DID
    '''
    def __init__(self, query, DID):
        self.query = query
        self.rank = 0
        self.DID = DID
        self.keyword_dict = {}
        self.unique_keyword_num = 0
        self.magnitude = 0
        self.similarity_score = []

    def show(self):
        print('For query: ')
        print(self.query)
        print('The top ' + str(self.rank) + ' result is: ')
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
        ret_result: list, contains top three high similarity score results.
    '''
    ret_result = []
    for query in processed_query:
        candidate_list = []
        for term in query:
            for k,v in inverted_file[term].posting_dict.items():
                if k not in candidate_list:
                    candidate_list.append(k)
        document_weight_vector = compute_weight_by_tfidf(query, inverted_file, document_list, candidate_list, total_sum)
        query_result_list = compute_all_similarity_result(query, document_weight_vector, candidate_list)
        query_result_list = sorted(query_result_list, key=lambda x: x.similarity_score, reverse=True)
        ret_result.append(query_result_list[:3])
    return ret_result


def compute_weight_by_tfidf(query, inverted_file, document_list, candidate_list, total_sum):
    '''
    Compute the tfidf value for every candidates in every query. The weight of each query word is 
    represented by weight = (tf / total_tf) * log(N / df + 1)
    Args:
        query: list, a list of query words
        inverted_file: list, used to compute df
        document_list: list, used to compute tf
        candidate_list: list, contains all the retrieved document ID in one query
        total_sum: int, used to compute tfidf
    Returns:
        document_weight_vector: list, each element is a list of float, 
                                representing a document vector for this candidate
    '''
    document_weight_vector = []
    for candidate in candidate_list:
        single_weight_vector = []
        for term in query:
            tf = 0
            if term in document_list[candidate].word_bag:
                tf = len(document_list[candidate].word_bag[term])/ total_sum
            df = len(inverted_file[term].posting_dict)
            tfidf = tf * math.log(len(document_list) / df+1)
            single_weight_vector.append(tfidf)
        document_weight_vector.append(single_weight_vector)
    return document_weight_vector


def compute_all_similarity_result(query, document_weight_vector, candidate_list):
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
    for _index, weight_vector in enumerate(document_weight_vector):
        query_weight = []
        for i in range(len(query)):
            query_weight.append(1)
        d_times_q = sum(weight_vector)
        mag_d = 0
        for num in weight_vector:
            mag_d += num ** 2
        mag_d = math.sqrt(mag_d)
        # mag_q = math.sqrt(len(query))
        cosine_score = d_times_q  # /(mag_d * mag_q)
        query_result = QueryResult(query, candidate_list[_index])
        query_result.similarity_score = cosine_score
        query_result.magnitude = mag_d
        query_result_list.append(query_result)
    return query_result_list


def complete_result(retrieve_results, document_list, inverted_file, total_sum):
    '''
    Complete the result. Since the QueryResult instance still have several term uncompleted.
    The five highest weighted keywords of the document and the posting lists is computed in compute_top5_keyword.
    The number of unique keywords in the document is computed in compute_unique_num.
    Args: 
        retrieve_results: list, contains top three high similarity score results.
        document_list: list, each element contains a Document instance 
        inverted_file: list, used to compute tfidf for indexing the key words
        total_sum: int, used to compute tfidf
    Returns:
        final_result: list, a list of completed QueryResult instance
    '''
    final_result = []
    for query_result in retrieve_results:
        for _index, single_result in enumerate(query_result):
            single_result.rank = _index + 1
            single_result.keyword_dict = compute_top5_keyword(single_result.DID, document_list, inverted_file, total_sum)
            single_result.unique_keyword_num = compute_unique_num(single_result.DID, document_list, inverted_file)
            final_result.append(single_result)
    return final_result


def compute_top5_keyword(DID, document_list, inverted_file, total_sum):
    '''
    Compute the top 5 keyword in one document based on the tfidf value
    Args: 
        DID: int, document ID
        document_list: list, used to compute tf
        inverted_file: list, used to compute df
        total_sum: int, used to compute tfidf
    Returns:
        ret_result: list, contains top 5 keyword and its posting list
    '''
    word_dict = {}
    ret_result = []
    document = document_list[DID]
    for k,v in document.word_bag.items():
        tf = len(v) / total_sum
        idf = math.log(len(document_list)/(len(inverted_file[k].posting_dict)+1))
        tfidf = tf * idf
        word_dict[k] = tfidf
    word_dict = sorted(list(word_dict),key=lambda x: x[0], reverse=True)
    word_dict = word_dict[:5]
    for word in word_dict:
        ret_result.append((word, inverted_file[word].posting_dict))
    return dict(ret_result)


def compute_unique_num(DID, document_list, inverted_file):
    '''
    Find the number of unique word in one document
    Args: 
        DID: int, document ID
        document_list: list, used to find document
        inverted_file: list, used to find whether unique
    Returns:
        unique_num: int, the number of unique word
    '''
    unique_num = 0
    document = document_list[DID]
    for word, _ in document.word_bag.items():
        if len(inverted_file[word].posting_dict) == 1:
            unique_num += 1
    return unique_num


if __name__ == "__main__":
    collection_data = read_data('collection-100.txt')
    query_data = read_data('query-10.txt')

    processed_data, total_sum = preprocess(collection_data)
    processed_query, _ = preprocess(query_data)

    document_list = generate_word_bag(processed_data)
    inverted_file = generate_inverted_file(document_list)

    retrieve_results = query_retrieve(processed_query, inverted_file, document_list, total_sum)

    final_result = complete_result(retrieve_results, document_list, inverted_file, total_sum)
    for result in final_result:
        result.show()

