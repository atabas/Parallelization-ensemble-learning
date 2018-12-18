# Parallelization of segments of the end to end ensemble training and evaluation for a text classification task

End to end heterogeneous scalable ensemble learning algorithm which demonstrates the performance gain of using parallelism in 
certain portions of the code. The ensemble learning algorithm is used for the training and evaluation of text classification.
To be specific, 3 shallow machine learning models (namely Logistic Regression, Naive Bayes and Random Forest) and 3 deep neural models 
(LSTMs with 3 different dropout rates) are trained and the training and evaluation of these models is executed in parallel. 
Aside from the actual training and evaluation, the preprocessing of the data is also parallelized. 

Because of lack of access to a distributed cluster (Hadoop/Spark) and GPU server, the implementation is limited to a single 4-core CPU 
machine using Python’s multiprocessing framework with the joblib wrapper. 
For the purpose of text classification, a problem where the task is to identify authors given text excerpts from books written by them has
been chosen. For this dataset, there are 3 predefined authors: Edgar Allan Poe, Mary Shelley, and HP Lovecraft.
The dataset from this Kaggle competition is from: https://www.kaggle.com/c/spooky-authoridentification . 
The training dataset contains more than 18,000 records which, while not gigantic, 
is big enough to allow us to simulate a big data problem and experiment with parallelization.

The end to end algorithm has been implemented with python’s multi-processing framework along with joblib wrapper, numpy, pandas, 
scikit learn and Keras.
