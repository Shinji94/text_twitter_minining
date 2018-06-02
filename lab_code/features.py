#!/usr/bin/env python3
 
import nltk
from nltk.corpus import stopwords

#Series of feature extraction methods which return NLTK's FreqDist (http://www.nltk.org/api/nltk.html?highlight=freqdist#nltk.probability.FreqDist),
#   which is based upon a Counter (https://docs.python.org/3/library/collections.html#collections.Counter),
#      which is based on a dict (https://docs.python.org/3/library/stdtypes.html#dict)
#Methods can take in tokens list, text file itself, or another fql.

#Takes in list of tokens, e.g. from tokeniser.
def count_tokens(tokens):
    return nltk.FreqDist(tokens) #produce frequency list by counting occurence of each token, make all lowercase.

#Basic word counter, will depend on tokeniser, this could filter for what should be considered a "word", e.g. using a regular expression filter.
def count_words(tokens):
    return count_tokens(t.lower() for t in tokens)

#Filters fql based on list, e.g. from top 500 terms in corpus.
def restrict_freq_dist(freq_dist, restricted_list):
    return nltk.FreqDist({t: freq_dist[t] for t in restricted_list}) #filter freq_dist by looking up restricted terms (t) in freq_dist. (uses dictionary comprehension.)

#Takes in FreqDist from count_tokens method (i.e. word list, which can be filtered), and list of function words.
def count_function_words(token_freq_dist, functionwords=stopwords.words('english')):
    return restrict_freq_dist(token_freq_dist,functionwords)

#for loading custom function words.
def read_function_words(file='functionwords.txt'):
    with open(file) as f:
        fws = []
        lines = f.readlines()
        for line in lines:
            fws.append(line.strip())
    return fws

#Takes in text file, extracts single characters, and counts these.
def count_chars(text):
    text_chars = list(text) #splits string into list of characters.
    return count_tokens(text_chars) #produce token count based on each token being a char.

#uses NLTK's pos tagger to tag token stream and count pos tags.
def count_pos(tokens):
    pos_tagged = nltk.pos_tag(tokens) #returns words paired with pos tags.
    just_pos = [tag[1] for tag in pos_tagged]
    return count_tokens(just_pos)

#Single features scores
#Could use tokens list, fql, or whole text.

def avg_word_length(tokens):
    return sum(len(token) for token in tokens) / len(tokens)
