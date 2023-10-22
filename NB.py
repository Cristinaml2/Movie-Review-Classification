# importing libraries
import numpy as np
import pandas as pd

# loading the small corpus data
small_corpus = pd.read_excel('small-corpus-data.xlsx')
X = small_corpus.iloc[:, 1:].values
y = small_corpus.loc[:, 'Label'].values

# loading the training data
train_data = pd.read_csv('training_data.csv')

# loading the testing data
test_data = pd.read_csv('testing_data.csv')

# features values of training data
X_train = train_data.iloc[:, 1:].values
# labels of training data
y_train = train_data.loc[:, 'Label'].values

# features values of testing data
X_test = test_data.iloc[:, 1:].values
# labels of testing data
y_test = test_data.loc[:, 'Label'].values

# naive bayes class
class NaiveBayes(object):
    
    # initialize object state
    def __init__(self, alpha = 1.0):
        
        self.prior = None
        self.word_counts = None
        self.lk_word = None
        self.alpha = alpha
        self.is_fitted = False
    
    # method to fit naive bayes on training data    
    def fit(self, X, y):
        
        n = X.shape[0]

        # each class (pos, neg) sub array
        X_by_class = np.array([X[y == c] for c in np.unique(y)])
        
        # prior probabilities of each class
        self.prior = np.array([len(X_class) / n for X_class in X_by_class])
        
        # each sub array word counts
        self.word_counts = np.array([X_class.sum(axis = 0) for X_class in X_by_class]) + self.alpha
        
        # word counts divided by the total number of times all words appear in each class
        self.lk_word = self.word_counts / self.word_counts.sum(axis = 1).reshape(-1, 1)

        # set is_fitted true
        self.is_fitted = True
        
        return self        
        
    # method to predict naive bayes probabilities on testing data 
    def predict_proba(self, X):
        
        # raise assertion-error if model is not fitted, it must be fit before predicting
        assert self.is_fitted
        
        class_numerators = np.zeros(shape = (X.shape[0], self.prior.shape[0]))
        
        # loop over each observation to calculate conditional probabilities
        for i, x in enumerate(X):
            
            word_exists = x.astype(bool)
            lk_words_present = self.lk_word[:, word_exists] ** x[word_exists]
            lk_review = (lk_words_present).prod(axis = 1)
            class_numerators[i] = lk_review * self.prior
        
        # likelihood of review across all classes
        normalize_term = class_numerators.sum(axis = 1).reshape(-1, 1)
        conditional_probas = class_numerators / normalize_term
        
        return conditional_probas    
        
    
    # method to predict class with highest probability
    def predict(self, X):
        
        return self.predict_proba(X).argmax(axis = 1)
        
# initializae the naive bayes instance
model = NaiveBayes(alpha = 1.0)             

# fit on the small corpus data except last record which is for test
model.fit(X[:-1], y[:-1])

# get prediction on the last record 
class_ = model.predict(X[-1].reshape(1, -1))
if class_ == 1:
   print('Predicted Class: Comedy')
else:
   print('Predicted Class: Action')
   
# get probability of each class for the last record 
probabilities = model.predict_proba(X[-1].reshape(1, -1))
print('Action Class Probability: ', probabilities[0][0])
print('Comedy Class Probability: ', probabilities[0][1])


# fit on the movie review training data
model.fit(X_train, y_train)

# get predictions on movie review testing data
y_pred = model.predict(X_test)

# accuracy score
accuracy = (y_test == y_pred).mean()

# add the predicted label column with accuracy as in last row in a dataframe
df = pd.DataFrame(np.append(y_pred, [accuracy], axis = 0), columns = ['Predicted Label'])
# join the test data actual labels and feature vector
df = df.join(test_data)
# write the results in a csv file
df.to_csv('predictions.csv', index = False)