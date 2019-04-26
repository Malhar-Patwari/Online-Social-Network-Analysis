# coding: utf-8

# Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO    
    tkn=[]
    for i in range(len(movies)):        
        tkn.append(tokenize_string(movies.genres[i]))
    movies['tokens'] = pd.Series(tkn, movies.index)       
    return movies    


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO   
    
    tok_list=[]    
    csr_list=[]    
    cnt = 0
    tf_val = 0    
    vocab=defaultdict(lambda:0)    
    freq_term_dataset=Counter()	
    freq_term_doc=Counter()
    for i in movies.tokens:
    	tok_list.extend(i)    
    for tok in sorted(set(tok_list)):    	
    	vocab[tok]=cnt
    	cnt+=1    
    
    for term in movies.tokens:
    	freq_term_dataset.update(term)    

    for num in range(len(movies)):
    	freq_term_doc.clear()
    	freq_term_doc.update(movies.tokens[num])
    	max_num = max(freq_term_doc.values())
    	sorted_doc_terms=sorted(set(movies.tokens[num]))
    	row=[]
    	col=[]
    	data=[]    	    	
    	for term in sorted_doc_terms:
    		row.append(0)
    		col.append(vocab[term])
    		tf_val = (freq_term_doc[term]/ max_num*(math.log10(len(movies)/freq_term_dataset[term])))
    		data.append(tf_val)    		   	    
    		x=csr_matrix((data, (row, col)), shape=(1, len(vocab)))    		
    	csr_list.append(csr_matrix((data, (row, col)), shape=(1, len(vocab))))    
    movies['features'] = pd.Series(csr_list, index=movies.index)    
    return tuple((movies, vocab))




def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    return (a.dot(b.T).toarray()[0][0]) / (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))    


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as thelen9
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    r_mid = defaultdict(lambda:0)
    r_ratings = defaultdict(lambda:0)    
    rating_list=[]
    for r in (ratings_train.index):    	
    	r_mid[r]=ratings_train.movieId[r]
    	r_ratings[r]=ratings_train.rating[r]


    for i in (ratings_test.index):
    	uid =  ratings_test.userId[i]
    	mid = ratings_test.movieId[i]    	
    	user_movie_index =[]
    	user_movie_index = ratings_train.movieId[ratings_train.userId == uid].index
    	a_csr = movies.loc[movies.movieId == mid].squeeze()['features']
    	flag = True
    	weighted_avg = 0
    	cos = 0
    	total_rates = 0
    	for j in user_movie_index:
    		b_csr = movies.loc[movies.movieId == r_mid[j]].squeeze()['features']
    		cos = cosine_sim(a_csr,b_csr)
    		if cos > 0:
    			weighted_avg += cos*r_ratings[j]
    			total_rates+=cos
    			flag = False
    	if flag == False:
    		rating_list.append(weighted_avg/total_rates)
    	else:	
    		rating_list.append(ratings_train.rating[ratings_train.userId == uid].mean())    	
    return np.array(rating_list)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():	
	download_data()
	path = 'ml-latest-small'
	ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
	movies = pd.read_csv(path + os.path.sep + 'movies.csv')    
	movies = tokenize(movies)
	movies, vocab = featurize(movies)	
	print('vocab:')
	print(sorted(vocab.items())[:10])
	ratings_train, ratings_test = train_test_split(ratings)
	print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))	
	predictions = make_predictions(movies, ratings_train, ratings_test)		
	print('error=%f' % mean_absolute_error(predictions, ratings_test))
	print(predictions[:10])		


if __name__ == '__main__':
    main()
