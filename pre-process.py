# importing libraries
import os
import re
import numpy as np
import pandas as pd

# positive reviews files from train directory
pos_files_train = os.listdir('movie-review-HW2/train/pos')

# negative reviews files from train directory
neg_files_train = os.listdir('movie-review-HW2/train/neg')

# positive reviews files from test directory
pos_files_test = os.listdir('movie-review-HW2/test/pos')

# negative reviews files from test directory
neg_files_test = os.listdir('movie-review-HW2/test/neg')


# method to read and combine data from directories
def combine(directory, pos_files, neg_files):
    
    # reviews data in a single list
    data = []
    
    # reviews labels in a single list
    labels = []
    
    # iterate for each positive review file
    for file in pos_files:
        # open the file in read mode from the directory
        with open(directory + 'pos/' + file, 'r', encoding = 'utf-8') as f:
            # read its content and save in the list
            data.append(f.read())
            
        # append '1' to represent positive review
        labels.append(1)
            
    # iterate for each negative review file
    for file in neg_files:
        # open the file in read mode from the directory
        with open(directory + 'neg/' +  file, 'r', encoding = 'utf-8') as f:
            # read its content and save in the list
            data.append(f.read())

        # append '0' to represent negative review
        labels.append(0)

    # return the data and their labels
    return data, labels


# train reviews data and labels in lists
train_data, train_labels = combine('movie-review-HW2/train/', pos_files_train, neg_files_train)
 
# test reviews data and labels in lists
test_data, test_labels = combine('movie-review-HW2/test/', pos_files_test, neg_files_test)


# method to preprocess the data
def preprocess(data):

    # list to store preprocess data
    preprocess_data = []
    
    # iterate for each review
    for review in data:
    
        # filter text from review by separating punctuations and extra white-spaces
        review = ' '.join(re.sub('[^a-zA-Z]', ' ', review).split())
        
        # converting in lower case
        review = review.lower()
        
        # append in the list
        preprocess_data.append(review)
    
    # return the preprocess data list
    return preprocess_data


# method to create bag of words features
def bow_features(data, vocab):
    
    # preprocess the data
    data = preprocess(data)
    
    # bow vector
    bow_vector = np.empty((0, len(vocab)))
    
    # iterate for each review
    for review in data:
        
        # words in the review
        words = review.split()
        
        # array to store features values
        bow = np.zeros(len(vocab))
        
        for word in words:
            
            for i, w in enumerate(vocab):
                # if word corresponds to vocabulary then increase its count
                if word == w:
                   bow[i] += 1 
            
        # add this review features values in bag of words vector
        bow_vector = np.append(bow_vector, [bow], axis = 0)

    # return bag of words vector
    return bow_vector

# load the vocabulary from the directory
with open('movie-review-HW2/imdb.vocab', 'r', encoding = 'utf-8') as f:
    vocab = f.read().split()

# bag of words feature vector for train data
train_bow = bow_features(train_data, vocab)
    
# bag of words feature vector for test data
test_bow = bow_features(test_data, vocab)


# save feature vector and labels of train data in a csv file
df1 = pd.DataFrame(train_bow)
df1.insert(loc = 0, column = 'Label', value = train_labels)
df1.to_csv('training_data.csv', index = False)

# save feature vector and labels of test data in a csv file
df2 = pd.DataFrame(test_bow)
df2.insert(loc = 0, column = 'Label', value = test_labels)
df2.to_csv('testing_data.csv', index = False)