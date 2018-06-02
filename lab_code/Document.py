#!/usr/bin/env python3
 
import tokeniser, features

class Document:

    def __init__(self, name, metadata = {}):
        self.name = name
        self.fqls = {} #frequency lists, in form of FreqDist
        self.features = {} #Single feature scores.
        self.metadata = metadata

    def process_document(self, text):
        #fws = features.read_function_words()
        tokens = tokeniser.tokenise(text)
        word_freqs = features.count_words(tokens)
        #fw_freqs = features.count_function_words(word_freqs,fws)
        #char_freqs = features.count_chars(text)
        #pos_freqs = features.count_pos(tokens)

        #fqls added here are available for the Document.
        #self.fqls['chars'] = char_freqs
        #self.fqls['fws'] = fw_freqs
        #self.fqls['pos'] = pos_freqs
        self.fqls['bow'] = word_freqs

        #single features added here are available for Document.
        #self.features['avg_word_length'] = features.avg_word_length(tokens)

    def process_document_from_textfile(self, textfile):
        with open(textfile,'r', encoding='UTF-8') as f:
            text = []
            for line in f:
                text.extend(f)
            self.process_document(text)

    #fql = frequency list name
    #term frequency for tfidf, currently set to relative frequency.
    def tf(self, fql, term):
        return self.fqls[fql].freq(term) #returns freq/N (N=total terms in document)

    #fql = frequency list named
    #returns 1 if term present in frequency list, 0 otherwise.
    def tbinary(self, fql, term):
        if self.fqls[fql][term]  > 0:
            return 1
        else:
            return 0
