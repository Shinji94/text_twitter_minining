code used in lab


still figuring out how to use other's git but code can be found here : https://github.com/barona586/adm


1. Setup
This week's code is available on a GitHub repo: https://github.com/barona586/adm

From the VM (or your own machine), clone the repo with:

$ git clone https://github.com/barona586/adm.git

Or, if already cloned previously, you can run:

$ git pull

from the repository, and you should get the new material.

Best to do this in your h-drive. You will find week14, week15, and week16 there too. For week20 you may need to install some python packages, these are listed in week20/requirements.txt. Simply run:

$ sudo pip install -r requirements.txt

Also there are nltk packages to download, this easiest down by running the supplied python script:

$ python3 install_nltk_packages.py

For the first part of the lab, you will be using the MP Twitter data you used previously. It is available to download and unzip from: https://modules.lancaster.ac.uk/pluginfile.php/1583319/mod_page/content/4/mps.zip  I can't put this on the GitHub repo unfortunately.

The Twitter data is the tweets from a series of labour and conservative MPs. It is the raw full_text from the Twitter API, you will probably want to clean it up and tokenise it sensibly (as per last previous labs) if you haven't already.

Document.py: added the ability to process a document from a text file, or raw text, and a binary term frequency method. Reduced feature set used to just bow.
Corpus-nltk.py: is an updated version of Corpus.py, with a new method for converting to a features usable for classification in NLTK.
analysis-nltk.py: This is where the classification happens, and also reading in the data and splitting into a training/test split.
The code is setup to run with a simple bag-of-words features, with NLTK's own inbuilt Naive Bayes classifier and Decision Tree classifier. Try running this:
$ python3 analysis_nltk.py

and then spend a little time trying to improve the classifier and observing the effects. Things you could try:

Adding different, or additional features, note these must be binary features (or "binned") to be used with the implementations of naive bayes and decision tree.
Changing the train/test split, and/or adding a validation set.
Using different tokenisation and pre-processing.

3. Classification with scikit learn (sklearn)
The classifiers within NLTK are limited. It is possible to export feature tables and run classification elsewhere (e.g. in R or Matlab), but a more common path, especially with language data, is to use scikit learn: http://scikit-learn.org/, which is a python library for machine learning. There are many online tutorial using sci-kit learn, and the API is quite good. I've included links in the code that you should look at as you go along.

Make sure you have pulled the latest code from the GitHub repo (or downloaded the zip of the code). You should find analysis_sklearn.py. This code uses sklearn's own libraries to complete basic feature extraction, and is a good place to start using sklearn. The only previous code it uses is tokeniser.py.

The code is setup to run a bag of words model with logistic regression, and TFIDF normalisation. Try this and observe the results.

$ python3 analysis_sklearn.py

and then spend a little time trying to improve the classifier and observing the effects. Things you could try:

Using different classifiers, e.g. SVM, RandomForest.
Changing the parameters of the machine learning classifiers.
Changing the parameters of the TFIDF normalisation, or using different normalisation.
Using different features, the "analyser" can easily be changed to char, or char_wb, and different ngram ranges can be included with ngram_range. See http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
Changing the train/test split, and/or adding a validation set.
Using different tokenisation and pre-processing.
