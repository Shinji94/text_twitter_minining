#!/usr/bin/env python3
 
import re
import sys
import nltk
import codecs
import matplotlib
nltk.download('punkt') #download tokeniser models, will download first time, and find on second run. You can remove this line once installed

def tokenise(text):
    return nltk_twitter_tokenise(text) #change this method to change the tokenisation used.

#The most simple white space tokeniser.
#Can you think of a better white space tokeniser, e.g. that splits on multiple spaces, or other white space characters?
#Can you come up with an alternative than using split, and instead using findall with a regular expression to do the same?
def whitespace_tokenise(text):
    return text.split(" ")

#Simple "word tokeniser", i.e. finds actual words.
#In what ways is this limited? How much can you improve it?
#Can you invert this, using a split on "non-word characters" instead?
def simple_match_tokenise(text):
    p = re.compile(r"[a-zA-Z]+")
    return p.findall(text)

#Here you can create your own tokeniser by searching for a series of patterns in turn.
#Python regular expressions uses a "Traditional Nondeterministic Finite Automaton", which is common for regular expressions.
#This means that for alternation, as used in this tokeniser, each option is tried sequentially (left to right), a longer match (i.e. "greedy") may be ignored if an earlier option leads to a successful match.
#It is therefore important to consider the order of the regular expressions, as a more general pattern may match some text before a more specific pattern has chance to see it.
#To demonstrate this, swap URL and '\w+' below in the patterns sequence.
#General hint, having a final catch all of "non-white-space" will pick up anything not explicity looked for earlier, but not always needed/wanted.
def custom_tokenise(text):
    URL = '(?:https?://)(?:[-\w]\.)+[a-zA-Z]{2,9}[-\w/#~:.?+=&%@~]*' #this is a simple URL pattern, more complicated that catch different URLs are possible.
    word = '\w+'
    patterns = (URL, word)
    joint_patterns = '|'.join(patterns)
    p = re.compile(r'(%s)' % joint_patterns)
    return p.findall(text)


def nltk_tokenise(text):
    return nltk.word_tokenize(text)

def nltk_twitter_tokenise(text):
    twtok = nltk.tokenize.TweetTokenizer()
    return twtok.tokenize(str(text))

#takes file, reads line by line, and returns list of tokens.
def tokenise_file(textfile):
    with open(textfile,'rb') as f:
        tokens = []
        for line in f:
#            line.decode("utf-8")           #change code here
            line_tokens = tokenise(line.strip())
            tokens.extend(line_tokens)
    return tokens

#perform basic frequency analysis with NLTK
def frequency_analysis(tokens,number = 50):
    freq = nltk.FreqDist(tokens) #produce frequency list by counting occurence of each token
    for key,val in freq.most_common(): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
        print(str(key) + ":" + str(val))

    freq.plot(number, cumulative=False) #makes frequency plot of top n tokens, using matplotlib. Try changing cumulative to True. Should produce nice Zipfian curve given enough tokens.


#entry point for running code.
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage %s textfile" % sys.argv[0])
        sys.exit(1)
    word = list()
    textfile = sys.argv[1] #pass in text file on command line.
    tokens = tokenise_file(textfile) #extract tokens from text, using tokenise_file, which calls a hard-coded tokenisation method.
    for token in tokens: #iterate tokens and print one per line. Could redirect this to file with > on command line, or code this directly.
        
        word.append(token)
    token = str(word)
    token = nltk_tokenise(str(word))
    number = int(input('input the number of words showed in graph'))
    frequency_analysis(word,number =number) 
    print("Total %d tokens" % len(tokens))
