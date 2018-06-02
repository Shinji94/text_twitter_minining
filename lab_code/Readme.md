code used in lab


still figuring out how to use other's git but code can be found here : https://github.com/barona586/adm


First you will continue from where you left off in Week 16. The base code for the last lab is provided, where you completed some feature extraction (feel free to use your own previous feature extractors), but be aware there have been some small updates to the code:

Document.py: added the ability to process a document from a text file, or raw text, and a binary term frequency method. Reduced feature set used to just bow.
Corpus-nltk.py: is an updated version of Corpus.py, with a new method for converting to a features usable for classification in NLTK.
analysis-nltk.py: This is where the classification happens, and also reading in the data and splitting into a training/test split.
The code is setup to run with a simple bag-of-words features, with NLTK's own inbuilt Naive Bayes classifier and Decision Tree classifier. Try running this:
$ python3 analysis_nltk.py

and then spend a little time trying to improve the classifier and observing the effects. Things you could try:

Adding different, or additional features, note these must be binary features (or "binned") to be used with the implementations of naive bayes and decision tree.
Changing the train/test split, and/or adding a validation set.
Using different tokenisation and pre-processing.
