#!/usr/bin/env python3
 
from Document import Document
from Corpus_nltk import Corpus
from os import listdir
from os.path import isfile, join, splitext, split
import random
import nltk
import numpy as np

import os

os.chdir('C:\\Users\\Hasee\\$USER\\scc-110')
def process_twitter_folder(folder, metadata):
    textfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".txt")]
    #textfiles = textfiles[:2] #limit for quick processing if you wish, but should be ok to work with all.
    documents = []
    for tf in textfiles:
        textname = splitext(split(tf)[1])[0] #extract just username from filename.
        print('Processing ' + textname)
        document = Document(textname, metadata)
        document.process_document_from_textfile(tf)
        documents.append(document)
    return documents

#documents = []
#process_twitter_folder("mps/conservative", {"party": "conservative"})
#textfiles = [join("mps/conservative", f) for f in listdir("mps/conservative") if isfile(join("mps/conservative", f)) and f.endswith(".txt")]
#documents = []
#for tf in textfiles:
#        textname = splitext(split(tf)[1])[0] #extract just username from filename.
#        print('Processing ' + textname)
#        document = Document(textname, {"party": "conservative"})
#        document.process_document_from_textfile(tf)
#        documents.append(document)
if __name__ == '__main__':
    documents = []
    documents.extend(process_twitter_folder("mps/conservative", {"party": "conservative"}))
    documents.extend(process_twitter_folder("mps/labour", {"party": "labour"}))
    
    #seed makes same train/text split each time, you may want to remove this real optimisation
    random.seed(13)
    random.shuffle(documents)
    
    n = len(documents)
    test_size = int(n/5)
#    test_size = int(n/7) #changing test set size

    train_docs = documents[:-test_size]
    test_docs = documents[-test_size:]
    
    print("Train:")
    print([doc.name for doc in train_docs])
    
    print("Test:")
    print([doc.name for doc in test_docs])
    
    train_corpus = Corpus()
    train_corpus.add_documents(train_docs)
    
    #simple bow, top 500 words across training data.
    bow_feats = train_corpus.restricted_fql_filter("bow", 500, 0)
    fql_feats = {"bow": bow_feats}
    sgl_feats = []
    train_set = train_corpus.featureset("party", fql_feats, sgl_feats)
    
    test_corpus = Corpus()
    test_corpus.add_documents(test_docs)
    #use same feature_names for test as used for training.
    test_set = test_corpus.featureset("party", fql_feats, sgl_feats)
    
    
    print()
    print("NLTK Naive Bayes Classifier") #simple naive bayes, only takes binary features (e.g. whether doc contains word)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Accuracy: ", nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features() #these features are most useful in prediction, i.e. feature more likely to appear in one class over another.
    
    print()
    print("NLTK Decision Tree Classifier")
    classifier = nltk.DecisionTreeClassifier.train(train_set) #Decision tree also only takes binary features.
    print("Accuracy: ", nltk.classify.accuracy(classifier, test_set))
    print(classifier) #prints the decision tree.
