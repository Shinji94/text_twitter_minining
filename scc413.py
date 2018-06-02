'''
    Apply Data Mining project code
    author: Wnag Xinji  32026312
    data:2018-05-22 15:49:02
    all the code are working under the support of official documents and code used in the lab and write by myself.   
    reference:   https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/  [1]
                 https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
    and line 235~238 about producing the corpus comes from this website. [1]
    
'''
import timeit
start = timeit.default_timer()
import os
import nltk
import json
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.stanford import StanfordTokenizer 
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from os import walk
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
import string
from nltk.tokenize import wordpunct_tokenize
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
    
def frequency_analysis(tokens,number = 50):
    i = 0
    freq = nltk.FreqDist(tokens) #produce frequency list by counting occurence of each token
    for key,val in freq.most_common(): #for each token with frequency, most_common() provides the tokens in frequency order, highest first.
        print(str(key) + ":" + str(val))
        i +=1
        if i == 40:
            freq.plot(number, cumulative=False) #makes frequency plot of top n tokens, using matplotlib. Try changing cumulative to True. Should produce nice Zipfian curve given enough tokens.
            break
def avg_word_length(tokens):
    return sum(len(token) for token in tokens) / len(tokens)
'''
these are tokenises used in the code
1:to use white space tokenizer change tokenizer in pipline as :whitespace_tokenise
2:to use twitter tokenizer change tokenizer in pipline as : tokenise
3: to use a default tokenizer change tokenizer in pipline as : nltk.word_tokenize
4:to use  word punctuation change tokenizer in pipline as :  wordpunct_tokenize
'''
tokenise =nltk.word_tokenize #default one

def tokenise(text):
    return nltk_twitter_tokenise(text)
def nltk_twitter_tokenise(text):
    twtok = nltk.tokenize.TweetTokenizer()
    return twtok.tokenize(text)
def whitespace_tokenise(text):
    return text.split(" ")


# design pipeline which you can change tokenizer with
'''you can use : 1.tokenise 2 .nltk.word_tokenize 3.wordpunct_tokenize 4.whitespace_tokenise
'''
def define_tokenizer(tokenizer = tokenise):  
    # using a linear svc model
    pipeline = Pipeline([
    #  using self produce vocabulary by change replace by  
    #    ('vect', tfidf_vectorizer),
        ('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer=wordpunct_tokenize,   #change tokenizer here
                                 max_features=300)),
        ('tfidf',TfidfTransformer(norm='l2')),  
                                       
        ('clf',  LinearSVC()), 
    ])
    
    
    #TRAINING  a logistic regression
    pipeline1 = Pipeline([
    #     ('vect', tfidf_vectorizer),
        ('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer=nltk.wordpunct_tokenize, #change tokenizer here
                                 max_features=500)),
        ('tfidf', TfidfTransformer(norm='l2')),  #
        ('clf',  LogisticRegression()), 
    ])
    
    #using a random forest  classifier
    pipeline2 = Pipeline([
    #    ('vect', tfidf_vectorizer),
        ('vect', CountVectorizer(analyzer='word', stop_words=stopwords.words('english') + list(string.punctuation), tokenizer=nltk.wordpunct_tokenize, #change tokenizer here
                                 max_features=500)),
        ('tfidf', TfidfTransformer(norm='l2')),  #Parameters can be set for Tfidf, e.g. using "sublinear_tf=True" can be effective, using the log of tf. http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
                                        #Other tranforms can be used, although tfidf is very common for NLP classification tasks, along with binary features. http://scikit-learn.org/stable/modules/preprocessing.html.
        ('clf',  RandomForestClassifier()), #Other classifiers could be used here from sklearn, see e.g. http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html. You can also tune the parameters of the classifier.
    ])
    return pipeline,pipeline1,pipeline2

def self_producedvectorizer():  
    pipeline = Pipeline([
        ('vect', tfidf_vectorizer),
        ('tfidf',TfidfTransformer(norm='l2')),  
                             
        ('clf',  LinearSVC()), 
    ])
    pipeline1 = Pipeline([
        ('vect', tfidf_vectorizer),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf',  LogisticRegression()), 
    ])
    pipeline2 = Pipeline([
        ('vect', tfidf_vectorizer),
        ('tfidf', TfidfTransformer(norm='l2')),  
        ('clf',  RandomForestClassifier()),
    ])
    return pipeline,pipeline1,pipeline2

# loading file
s = 'C:\\Users\\Hasee\\Desktop\\dataset\\gb-celebs'
print('start loading...')
os.chdir(s)
#fetch file name
file_names = []
for (dirpath,dirnames,filenames) in walk(s):
    file_names.extend(filenames)

#randomly shuffle every time
random.shuffle(file_names)

#initilize our data
age = []
age_range = []
user_id = []
language = []
handle = []
gender = []
tweets = []
full_text = []
tokens = []
full_text = []
remove = []
file_number = 0

for file_name in file_names:
    with open(file_name) as f:
        data = json.load(f)
        json_dict = json.dumps(data)
    #dump values in the list and vonvert into dict
        age.extend([data['age']])
        age_range.extend([data['age_range']]) 
        user_id.extend([data['user_id']])
        language.extend([data['english']])
        gender.extend([data['gender']])
        handle.extend([data['handle']])
        #create list of tweets for user
        user_tweet = []
        text = []
        time = []
        true_text = []
        for x in range (len(data['tweets'])):
            time.extend([data['tweets'][x]['time']])
            #use the lower case of all the text content so it will not be count as two words if it appear twice
            text.extend([data['tweets'][x]['text'].lower()])
        true_text = ' '.join(text)
        #REMOVE the http links quoted in the text because it doess not provide any infomation
        remove = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', true_text, flags=re.MULTILINE)
        full_text = full_text + [remove]
        '''
        change the token function here to change the way of token :-b
        '''
#        nltk_twitter_tokenise()
        token = nltk.word_tokenize(remove)
            #change to change tokenizer here
        tokens.extend([token])
        file_number += 1 
        print('pocessing ',file_name)
        f.close()
full = ' '.join(full_text)

#create dataset
df = pd.DataFrame({'user_id':user_id,'age':age,'age_range':age_range,
                   'language':language,'handle':handle,'gender':gender,
                   'token':tokens,'text ':full_text})

toke = tokenise(str(full)) #twitter
frequency_analysis(toke,number =60)
avg_word_length(toke)
print('average token length is :',round(avg_word_length(toke),3))
print("Total %d tokens" % len(toke))

'''

These are code for differnt tokenizer in frequency analysis

toke = whitespace_tokenise(str(full))
frequency_analysis(toke,number =60)
avg_word_length(toke)
print('average token length is :',round(avg_word_length(toke),3))
print("Total %d tokens" % len(toke))

toke = nltk.wordpunct_tokenize(str(full))
frequency_analysis(toke,number =60)
avg_word_length(toke)
print('average token length is :',round(avg_word_length(toke),3))
print("Total %d tokens" % len(toke))

'''
#model part  kill the unknow data  as the number of female is smaller than male. just simply asert those unknow to female to balance the data.
label = [] 
for each_gender in gender:
    if each_gender == 'male':
        label.append(1)
    if each_gender =='female':
        label.append(0)
    if each_gender == 'unknow':
        label.append(0)
len(label)

#add  label to the dataset 
df['label'] = label

#setting my own vocabulary base on the tweets which is cleaned
t = [word for word in toke if word.isalpha()]
stop_words = set(stopwords.words('english'))
t = [w for w in t if not w in stop_words]
#we dont want those 'symbols' and the emoji in the text to creat to much noise
t = [word for word in t if len(word) > 1]
t1 = {}.fromkeys(t).keys()

vocabulary  = t1

print( 'avg_word length is : ',avg_word_length(t1))
print('length of my vocabulary embedding is ',len(t1))
tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)  
real_vec = tfidf_vectorizer.fit_transform(full_text) 

#avg accuracy using 10-cross validation
n = len(file_names)
test_size = int(n/5)

avg_1 = []  # to store accuracy 
avg_2 = []
avg_3 = []
pipeline,pipeline1,pipeline2 = define_tokenizer(tokenizer = tokenise)
#pipeline,pipeline1,pipeline2 = self_producedvectorizer  #just uncomment this one to use selfproduce vectorizer
for i in range(10):
    c = list(zip(full_text, label))
    random.shuffle(c)
    full_text,label = zip(*c)
    
    train_texts = full_text[:-test_size]
    train_labels = label[:-test_size]
    
    pipeline.fit(train_texts, train_labels)
    pipeline1.fit(train_texts, train_labels)
    pipeline2.fit(train_texts, train_labels)
    
    predictions = pipeline.predict(full_text[(len(file_names)-test_size):-1])
    predictions1 = pipeline1.predict(full_text[(len(file_names)-test_size):-1])
    predictions3 = pipeline2.predict(full_text[(len(file_names)-test_size):-1])
    avg_1.append(accuracy_score(label[len(file_names)-test_size:-1], predictions))
    avg_2.append(accuracy_score(label[len(file_names)-test_size:-1], predictions1))
    avg_3.append(accuracy_score(label[len(file_names)-test_size:-1], predictions3))
    
print('the avg accuracy of linear SVC is',sum(avg_1)/len(avg_1))
print('linear svc')
print(classification_report(label[len(file_names)-test_size:-1], predictions))

print('the avg accuracy of LR is',sum(avg_2)/len(avg_2))
print('logistic classifier')
print(classification_report(label[len(file_names)-test_size:-1], predictions1))

print('the avg accuracy of RF is',sum(avg_3)/len(avg_3))
print('random forest')
print(classification_report(label[len(file_names)-test_size:-1], predictions3))

plot_coefficients(pipeline.named_steps['clf'], pipeline.named_steps['vect'].get_feature_names())
stop = timeit.default_timer()
print("Run time: " + str(int(stop - start)) + " seconds.")