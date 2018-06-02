#!/usr/bin/env python3
  
import tokeniser
from os import listdir
from os.path import isfile, join, splitext, split
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import string
from nltk.corpus import stopwords

def extract_text(folder):
    textfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".txt")]
    texts = []
    for tf in textfiles:
        with open(tf,'r', encoding='UTF-8') as f:
            texts = []
            for line in f:
                texts.extend(f)
    return texts


#adapted from: https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(15, 5))
    colors = ["blue" if c < 0 else "red" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()

if __name__ == '__main__':
    con_texts = extract_text("mps/conservative")
    lab_texts = extract_text("mps/labour")
    texts = con_texts + lab_texts
    labels = ["Conservative"] * len(con_texts) + ["Labour"] * len(lab_texts)
    
    #split the data up into train and test splits.
    #Here a 4:1 split of the data is made, i.e. 20% of the data is used for testing, 80% for training. With bigger datasets, you would normally want to reduce the test size to give you more training data.
    #This must be randomised so you get a random sample of training data, we also stratify the sample to make it have the same distribution as the overall dataset. This is especially important for smaller datasets, where by chance you could get very unbalanced test/train sets.
    #We set a random seed to have the same random sample each time, but in practice if optimising features and model parameters, you probably want a different sample each time, so not to overfit.
    #Also, you can split off a separate dev set (e.g. by splitting train again), and hold out the test set until you've optimised.
    #You can also use cross validation.
    #http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.20, random_state = 413, stratify = labels)
    
    print("train: ", train_labels)
    print("test: ", test_labels)
    
    pipeline = Pipeline([
        #The count vectorizer takes in raw texts and produces a vector of counts for the document. It is used for bag of words, but can also capture word n-grams and character n-grams. Custom tokenisers can be used also, along with preprocessing, and plenty of other options: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        #Here we have a bag of words model, we use a custom tokeniser (whatever is set in the tokeniser class), remove stop words and punctuation (this could be done at the tokenisation stage too and/or preprocessing), and we limit to the top 500 words found in the training set.
        ('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer=tokeniser.tokenise, max_features=500)),
        ('tfidf', TfidfTransformer(norm='l2')),  #Parameters can be set for Tfidf, e.g. using "sublinear_tf=True" can be effective, using the log of tf. http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
                                        #Other tranforms can be used, although tfidf is very common for NLP classification tasks, along with binary features. http://scikit-learn.org/stable/modules/preprocessing.html.
        ('clf',  LinearSVC()), #Other classifiers could be used here from sklearn, see e.g. http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html. You can also tune the parameters of the classifier.
    ])
    
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)
    print("Predictions :", predictions)
    
    #Show results of classification, other options available: http://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
    print("Accuracy: ", accuracy_score(test_labels, predictions))
    print (classification_report(test_labels, predictions))
    
    
    #Will create plot with coefficients from the classification model, the top 20 indicating each class. How the coefficients are calculated will depend on the machine learning algorith used.
    plot_coefficients(pipeline.named_steps['clf'], pipeline.named_steps['vect'].get_feature_names())
