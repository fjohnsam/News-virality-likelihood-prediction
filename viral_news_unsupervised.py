import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import time
from googletrans import Translator
translator=Translator()
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer
tokeniser = TweetTokenizer()

######### Crawling News Websites ###########

#Hindustan Times (English lingual newspaper)

print("Crawled News from Hindustan Times\n")

url = "https://www.hindustantimes.com/topic/india"

r1 = requests.get(url)
r1.status_code

coverpage = r1.content

soup1 = BeautifulSoup(coverpage, 'html5lib')

coverpage_news = soup1.find_all(class_='media-body')

number_of_articles = 10
news_contents = []
list_links = []
list_titles1 = []

for n in np.arange(0, number_of_articles):

    # Getting the title
    title = coverpage_news[n].find('a')['title']
    list_titles1.append(title)


# Dainik Jagran (Hindi Lingual Newspaper)

print("\nCrawled News from Dainik Bhaskar\n")

url = "https://www.bhaskar.com/national/5"

r1 = requests.get(url)
r1.status_code

coverpage = r1.content

soup1 = BeautifulSoup(coverpage, 'html5lib')

coverpage_news = soup1.find_all(class_='list_topdata')

number_of_articles = 10
news_contents = []
list_links = []
list_titles2 = []

for n in np.arange(0, number_of_articles):

    # Getting the title
    title = coverpage_news[n].find('a')['title']
    list_titles2.append(title)

translated = translator.translate(list_titles2, src='hi', dest='en')
listtitles2=[]
for trans in translated:
	listtitles2.append(trans.text)

####### ### Virality detection ############

#Hyperparameters
minimum_similarity_score = 0.2
minimum_number_of_occurences_for_virality = 2 

# Combining News Titles from all Crawled News Sources
sentences = []
for i in np.arange(0, len(list_titles1)):
	sentences.append(list_titles1[i])
for i in np.arange(0, len(listtitles2)):
	sentences.append(listtitles2[i])

tokenisedsentences = []
for i in np.arange(0, len(sentences)):
	tokenisedsentences.append(tokeniser.tokenize(sentences[i]))

# Calculating Doc-frequency for words
DF = {}

for i in range(len(tokenisedsentences)):
    tokens = tokenisedsentences[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

for i in DF:
    DF[i] = len(DF[i])

#Calculating tf-idf

doc = 0

tf_idf = {}

for i in range(len(tokenisedsentences)):
    
    tokens = tokenisedsentences[i]

    counter = {}
    uniquetokens = []
    for x in tokens:
        if x not in uniquetokens:
            uniquetokens.append(x)
    for w in uniquetokens:
       counter[w]=tokens.count(w)

    words_count = len(tokens)

    for token in np.unique(tokens):
        
        tf = counter[token]/words_count

        try:
            df=DF[token]
        except:
            df=0

        idf = np.log((len(tokenisedsentences)+1)/(df+1)) #numerator is added 1 to avoid negative values
        
        tf_idf[doc, token] = tf*idf

    doc += 1

#Vectorising News titles

total_vocab = [x for x in DF]

D = np.zeros((len(tokenisedsentences), len(DF)))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

count = 0
score = 0

#Measuring Virality of news
print("\n\n\nNews with a score for being viral and prediction judgement of being viral/non-viral::\n")

for i,a in enumerate(D):
    count = 0
    score = 0
    for b in D:
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        if cos_sim > minimum_similarity_score and cos_sim < 1 :
            score+=cos_sim
            count+=1
    if count >= minimum_number_of_occurences_for_virality :
        print(sentences[i],"  ; viral score=",score,": viral\n\n")
    else:
        print(sentences[i],"  ; viral score=",score,"   : not viral\n\n")