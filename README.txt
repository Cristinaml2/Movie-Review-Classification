- After unzipping the file, use the folder 'Movie Review Classification' as your current working directory in Spyder or Python IDE that you have. Otherwise, 
  you will not be able to load the data files.


- Make sure you are using the directory structure same as attached because there is a use of this directory structure when reading the data files from train and 
  test directory.


- The folder 'movie-review' is the train and test data reviews text files and vocabulary used in bag of words features. Keep this same. Please don't modify this.


- The file 'preprocess.py' is the Python file. It will load the positive, negative reviews text files from the train and test directory, applies the preprocessing, 
  create the bag of words and save the results in CSV files with name 'training_data.csv' for train data and 'testing_data.csv' for test data. These CSV files will 
  be created automatically in the same directory  i.e. 'Movie Review Classification'.


- The file 'NB.py' is the Python file. It will load the CSV training and testing files (created in previous part) from the the same directory  
  i.e. 'Movie Review Classification', applies the Naive Bayes model and output the results in a CSV file with name 'predicitions.csv' in the same directory. The 
  numeric label for class 'Positive' is '1' and for class 'Negative' is '0'.
  
  This file also load the small corpus from a file 'small-corpus-data.xlsx', applies NB on it and print the last record output on Console. The numeric label for 
  class 'Action' is '0' and for class 'Comedy' is '1'. 
