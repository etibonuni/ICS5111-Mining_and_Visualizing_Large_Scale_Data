"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to 3 consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

twenty_train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

#tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
#X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

docs_new = ["God is love", "OpenGL on the GPU is fast", "Cancer rates are increasing"]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print("%r => %s" % (doc, twenty_train.target_names[category]))
    
text_clf = Pipeline([("vect", CountVectorizer()),
                      ("tfidf", TfidfTransformer()),
                      ("clf", MultinomialNB())])
text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset="test", categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)

text_log_clf = Pipeline([("vect", CountVectorizer()),
                      ("tfidf", TfidfTransformer()),
                      ("clf", LogisticRegression(penalty="l1", C=0.1))])
text_log_clf.fit(twenty_train.data, twenty_train.target)

predicted2 = text_log_clf.predict(docs_test)

text_knn_clf = Pipeline([("vect", CountVectorizer()),
                      ("tfidf", TfidfTransformer()),
                      ("clf", KNeighborsClassifier(weights="uniform", metric="euclidean"))])
text_knn_clf.fit(twenty_train.data, twenty_train.target)

predicted3 = text_knn_clf.predict(docs_test)
   # The training data folder must be passed as first argument
#languages_data_folder = sys.argv[1]
#dataset = load_files(languages_data_folder)

# Split the dataset in training and test set:
#docs_train, docs_test, y_train, y_test = train_test_split(
#    dataset.data, dataset.target, test_size=0.5)


# TASK: Build a vectorizer that splits strings into sequence of 1 to 3
# characters instead of word tokens

# TASK: Build a vectorizer / classifier pipeline using the previous analyzer
# the pipeline instance should stored in a variable named clf

# TASK: Fit the pipeline on the training set

# TASK: Predict the outcome on the testing set in a variable named y_predicted

# Print the classification report
#print(metrics.classification_report(y_test, y_predicted,
#                                    target_names=dataset.target_names))

# Plot the confusion matrix
#cm = metrics.confusion_matrix(y_test, y_predicted)
#print(cm)

#import matplotlib.pyplot as plt
#plt.matshow(cm, cmap=plt.cm.jet)
#plt.show()

# Predict the result on some short new sentences:
#sentences = [
#    u'This is a language detection test.',
#    u'Ceci est un test de d\xe9tection de la langue.',
#    u'Dies ist ein Test, um die Sprache zu erkennen.',
#]
#predicted = clf.predict(sentences)

#for s, p in zip(sentences, predicted):
#    print(u'The language of "%s" is "%s"' % (s, dataset.target_names[p]))
