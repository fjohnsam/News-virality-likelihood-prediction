# News-virality-likelihood-prediction

Library Requirements :
  1. pip install numpy
  2. pip install requests
  3. pip install BeautifulSoap
  4. pip install googletrans
  5. pip install gensim
  6. pip install nltk
  
Methodology :
1. Crawled current national news from English news website (https://www.hindustantimes.com)
2. Crawled current national news from hindi news website (https://www.dainikbhaskar.com)
3. Converted all crawled news to a common language ( hindi to english in this case )
4. Created a tf-idf score for all the crawled news.
5. a) Unsupervised Learning approach {implemented as of now} Steps: 
     i) Measured the similarity of a news with all other crawled news.
    ii)Hence clustered similar news.
    iii)If a cluster contains more than a threshold(hyperparameter) of news then it is predicted to be viral.
   b) Supervised Learning approach {Not implemented as of now} Steps: i) Train a GRU model with news collection from open-source repository. ii) Feed the crawled news to get the prediction
   
Unsupervised Learning is a better approach for measuring the virality of news as supervised approach uses news datasets from prebuilt repositories which contains older news, and thus not a good measure for predicting virality of current news articles.{as per my intuition}

Note:
You may set the Hyper-parameters:
1. minimum_similarity_score = 0.15 (default)
2. minimum_number_of_occurences_for_virality = 2 (default) 

to get appropriate results
