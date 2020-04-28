# News-virality-likelihood-prediction

Library Requirements :
  pip install numpy
  pip install requests
  pip install BeautifulSoap
  pip install googletrans
  pip install gensim
  pip install nltk
  
Methodology :
1. Crawled current national news from English news website (https://www.hindustantimes.com)
2. Crawled current national news from hindi news website (https://www.dainikbhaskar.com)
3. Converted all crawled news to a common language ( hindi to english in this case )
4. Created a tf-idf score for all the crawled news.
5. a) Unsupervised Learning approach {implemented as of now} Steps: 
     i) Measured the similarity of a news with all other crawled news.
    ii)Hence clustered similar news.
    iii)If a cluster contains more than a threshold(hyperparameter) of news then it is predicted to be viral.
   b) Train a GRU model with news collection from open-source repository. Feed the crawled news to get the prediction{Not implemented as of now}
   
Unsupervised Learning is a better approach for measuring the virality of news as supervised approach uses news datasets from prebuilt repositories which contains older news, and thus not a good measure for predicting virality of current news articles.{as per my intuition}
