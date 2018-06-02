#!/usr/bin/env python3
 
import Document
from nltk import FreqDist
import math

class Corpus:

    def __init__(self):
        self.fql_totals = {} #summed FreqDists
        self.fql_num_docs = {} #how many texts each fql item occurs, e.g. for idf.
        self.feature_lists = {} #lists of single feature scores.
        self.documents = [] #list of all texts in corpus

    def featureset(self, label_name, fql_features, single_features):
        featureset = []
        for doc in self.documents:
            features = {}
            for fql_name, feats in fql_features.items():
                for f in feats:
                    feat = "{}_{}".format(fql_name, f) #include fql_name so no chance of repeat of feature names (though you should still be careful of "double counting")
                    val = doc.tbinary(fql_name,f) #using binary features, for naive bayes
                    features[feat] = val
#            for single_feature in single_features:
#                features[single_feature] = doc.features[single_feature]
#              Not currently used, single features would need to be reduced to binary features (e.g. using "binning")

            label = doc.metadata[label_name]
            featureset.append((features, label))
        return featureset

    #adds to freqLists and features from Document instance.
    def add_document(self, doc):
        for key, doc_fql in doc.fqls.items():
            if key in self.fql_totals:
                self.fql_totals[key].update(doc_fql)
                self.fql_num_docs[key].update((+doc_fql).keys()) #(+doc_fql) is only positive values
            else:
                self.fql_totals[key] = doc_fql.copy()
                self.fql_num_docs[key] = FreqDist((+doc_fql).keys()) #(+doc_fql) is only positive values

        for key, doc_feature in doc.features.items():
            if key not in self.feature_lists:
                self.feature_lists[key] = []
            self.feature_lists[key].append(doc_feature)

        self.documents.append(doc)
    def add_documents(self, docs):
        for doc in docs:
            self.add_document(doc)
    #provides a list of fql terms from the named list (fql) to filter by. Can choose top x features, with minimum frequency y.
    def restricted_fql_filter(self, fql, top=0, min_freq=0):
        if top > 0:
            restricted_list = dict(self.fql_totals[fql].most_common(top))
        else:
            restricted_list = self.fql_totals[fql]
        if min_freq > 0:
            restricted_list = { term : restricted_list[term] for term in restricted_list if restricted_list[term] >= min_freq }
        return restricted_list.keys()
    #fql = frequency list name
    def idf(self, fql, term):
        return math.log(len(self.documents) / (1 + self.fql_num_docs[fql][term]))
