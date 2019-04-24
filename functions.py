import requests
import nltk
import json
import time
import pandas as pd
import re
import praw
import datetime as dt
import pytz
from tzlocal import get_localzone
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from scipy import stats
import numpy as np
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import ADJ, ADJ_SAT, ADV, NOUN
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import statsmodels
from statsmodels.stats import power
from numpy import mean
from numpy import var
from math import sqrt
import pymongo
import itertools
from pymongo import errors
from pymongo.errors import InvalidDocument
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pymongo import MongoClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import prawcore
import sklearn.metrics
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
lem = WordNetLemmatizer()
stem = PorterStemmer()


abs_word_list = ['absolutely', 'all', 'always', 'complete', 'completely', 'constant', 'constantly', 'definitely', 
                 'entire', 'ever', 'every', 'everyone', 'everything', 'full', 'must', 'never', 'nothing', 
                 'totally', 'whole']

stem_abs = ['absolut', 'all','alway', 'complet','complet','constant','constantli','definit','entir',
 'ever',
 'everi',
 'everyon',
 'everyth',
 'full',
 'must',
 'never',
 'noth',
 'total',
 'whole']

def connect_to_reddit():
    with open("./reddit_perms.json", "r") as f:
        reddit_information = json.load(f)
        connect_to_reddit = praw.Reddit(client_id=reddit_information["client_id"],
                            client_secret=reddit_information["client_secret"],
                            user_agent=reddit_information["user_agent"],
                            username=reddit_information["username"],
                            password=reddit_information["password"])
    print(connect_to_reddit.user.me(), 'connecting to reddit API')
    return connect_to_reddit
print('reddit connector created')

reddit = connect_to_reddit()

def connect_to_mongo(host='localhost', port=27017):
    connector = MongoClient(host, port)
    print('now connected to mongo')
    return connector
print('mongo connector now created')

def get_stem_abs(word_list):
    stem = PorterStemmer()
    stem_abs = []
    
    for word in word_list:
        stem_abs.append(stem.stem(word))

    return stem_abs
print('get stem abs list function created')

def get_lem_abs(word_list):
    lem = WordNetLemmatizer()
    lem_abs = []
    for word in word_list:
        lem_abs.append(lem.lem(word))
    return lem_abs
print('get lem abs list function created')

def get_stop_words(include_list, exclude_list, input_stop_words):
    print('stop words list length is now:')
    print(len(input_stop_words))
    print('adding words from include list')
    stop_words_new = input_stop_words + include_list
    print('new words have been added, stop words list length is now:')
    print(len(stop_words_new))
    print('removing exlude words')
    for i in stop_words_new:
        if i in exclude_list:
            
            stop_words_new.remove(i)    

    print('exclude list words removed. stop words length is now:')
    print(len(stop_words_new))
    return stop_words_new
print("get stop words function has been created")

def text_cleaner(string, stop_words):

    text = re.sub('[^a-zA-Z]', ' ', string)
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    text = " ".join(text)
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    words = [stem.stem(word) for word in filtered_sentence]
    sentence = " ".join(words)
    return sentence
print('text cleaner function created')

def get_date(submission):
    time = submission.created
    return dt.datetime.fromtimestamp(time)
print('get date function created')

def submissions_to_mongo(subreddit_list, mongo_db, reddit_connector, moderators, stop_words):
    for sub in subreddit_list:
        subreddit_dict= {}    
        for submission in reddit_connector.subreddit(sub).new(limit=500):
            if submission.author == None:
                pass
            elif submission.author.name in moderators:
                pass
            elif submission.selftext == '':
                pass
            elif submission.banned_by != None:
                pass
            elif len([i for i in (text_cleaner(submission.selftext, stop_words))])<10:
                pass

            else:

                subreddit_dict["created_utc"] = submission.created_utc
                subreddit_dict["created_at"] = get_date(submission)
                subreddit_dict["subreddit"] = submission.subreddit.display_name
                subreddit_dict["author"] = submission.author.name
                subreddit_dict["_id"] = submission.id
                subreddit_dict["selftext"] = submission.title + ' ' + submission.selftext
                subreddit_dict["cleaned_text"] = text_cleaner(submission.selftext, stop_words)
                subreddit_dict['percent_abs_words'] = round(get_perc_abs(clean_text(submission.selftext, stop_words), stem_abs),2)
                subreddit_dict["sentiment"] = round(SentimentIntensityAnalyzer().polarity_scores(submission.selftext)['compound'], 2)
                subreddit_dict["subjectivity"] = round(TextBlob(submission.selftext).sentiment.subjectivity,2)
                subreddit_dict["post_length"] = len(submission.selftext)
                subreddit_dict["hour_posted"] = get_date(submission).hour
                try:
                    mongo_db.insert_one(subreddit_dict)
                except Exception as e:
                    print("Could not insert text")
                    print("-"*20)
    df = mongo_db
    df = pd.DataFrame(list(df.find()))
    return df
    print('all finished')
print('submissions to mongo function created')

def clean_df(df, column='selftext', subreddit_col='subreddit', subreddit=None, stop_words=None):
    text_list=[]    
    corpus =[]
    for index, row in df.iterrows():
        if row[subreddit_col] != subreddit:
            pass
        else:
            text_list.append(row[column])
    
            for i in range(0, len(text_list)):
    #Remove punctuations
                text = re.sub('[^a-zA-Z]', ' ', text_list[i])
    
    #Convert to lowercase
                text = text.lower()
    
    #remove tags
                text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
                text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
                text = text.split()
    
    ##Stemming
                text = [stem.stem(word) for word in text if not word in stop_words]
                text = " ".join(text)
        
                corpus.append(text)
    return corpus
print('clean df function created')

def get_tot_word(corpus, stem_abs=None, n=None):
    abs_words = []
    other_words = []
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    for i in words_freq:
        if i[0] in stem_abs:
            abs_words.append(i)
        else:
            other_words.append(i)
    dict_abs = dict(abs_words)
    dict_other = dict(other_words)
    tot_abs = sum(dict_abs.values())
    tot_words = (sum(dict_abs.values())+sum(dict_other.values()))
    tot_other = sum(dict_other.values())
    percentage = ((tot_abs/tot_words)*100)
    return tot_words
print('get total word count function created')

def get_abs_word_count(corpus, stem_abs=None, n=None):
    abs_words = []
    other_words = []
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    for i in words_freq:
        if i[0] in stem_abs:
            abs_words.append(i)
        else:
            other_words.append(i)
    dict_abs = dict(abs_words)
    dict_other = dict(other_words)
    tot_abs = sum(dict_abs.values())
    tot_words = (sum(dict_abs.values())+sum(dict_other.values()))
    tot_other = sum(dict_other.values())
    percentage = ((tot_abs/tot_words)*100)
    return tot_abs
print('get absolutist word count created')

def get_perc_abs(corpus, stem_abs=None, n=None):
    abs_words = []
    other_words = []
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    for i in words_freq:
        if i[0] in stem_abs:
            abs_words.append(i)
        else:
            other_words.append(i)
    dict_abs = dict(abs_words)
    dict_other = dict(other_words)
    tot_abs = sum(dict_abs.values())
    tot_words = (sum(dict_abs.values())+sum(dict_other.values()))
    tot_other = sum(dict_other.values())
    percentage = ((tot_abs/tot_words)*100)
    return percentage
print('get percentage of absolutist words function created')

def get_abs_chart(corpus, stem_abs=None, n=None):
    abs_words = []
    other_words = []
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    for i in words_freq:
        if i[0] in stem_abs:
            abs_words.append(i)
        else:
            other_words.append(i)
            
    freq_dict = dict(abs_words)
    plt.figure(figsize=(15,10))
    plt.bar(range(len(freq_dict)), list(freq_dict.values()), align='center')
    plt.xticks(range(len(freq_dict)), list(freq_dict.keys()))
    plt.ylim(0,500000)
# # for python 2.x:
# plt.bar(range(len(D)), D.values(), align='center')  # python 2.x
# plt.xticks(range(len(D)), D.keys())  # in python 2.x

    plt.show()
print('get absolitist word count chart function created')

def get_post_distribution(df, subreddit_column='subreddit', subreddit=None, stem_abs=None):
    abs_perc_list = []
    for index, row in df.iterrows():
        if row[subreddit_column] == subreddit:
            entry = row['selftext']
            entry = entry.split()
            perc = get_perc_abs(entry, stem_abs=stem_abs)
            abs_perc_list.append(perc)
    plt.hist(abs_perc_list, bins=12,)
    plt.title('Distribution of absolutist words in {} forum'.format(subreddit))
    plt.xlabel('Percentage of Absolutist Words Used Per Post')
    plt.ylabel('Number of Posts')
    plt.xlim(-2, 9)
    plt.ylim(0,600)
    plt.show()
            
    return abs_perc_list
print('get post distribution function created')

def wordcloud(corpus, stop_words):
    wordcloud = WordCloud(collocations=False, background_color='white',stopwords=stop_words,max_words=20,max_font_size=75).generate(str(corpus))
    fig = plt.figure(1);
    plt.imshow(wordcloud);
    plt.axis('off');
    plt.show();
print('get word cloud function created')

def clean_text(string, stop_words=None):

    text = re.sub('[^a-zA-Z]', ' ', string)
    text = text.lower()
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    text = " ".join(text)
    word_tokens = word_tokenize(text)  
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    words = [stem.stem(word) for word in filtered_sentence]
    return words
print('get clean text function created')

def get_distribution(stem_abs):
    entry = submission.selftext
    entry = entry.split()
    perc = get_perc_abs(entry, stem_abs=stem_abs)
    return perc
print('get perc distribution created')

def get_word_freq(corpus=None, n=None):
    word_freq = pd.Series(' '.join(corpus).split()).value_counts()[:n]
    return word_freq
print('get word frequency function created')

def cohend(d1, d2):
    
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return (u1 - u2) / s
print('Cohens D function created')

def mongo_to_df(df, mongo_db):
        df = mongo_db
        df = pd.DataFrame(list(df.find()))
        return df
print('mongo to df function created')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
print('plot confusion matrix function created')

def evaluate(ytrain, yhattrain, ytest, yhattest, class_names):
    print('Training Precision: ', precision_score(ytrain, yhattrain))
    print('Testing Precision: ', precision_score(ytest, yhattest))
    print('\n\n')

    print('Training Recall: ', recall_score(ytrain, yhattrain))
    print('Testing Recall: ', recall_score(ytest, yhattest))
    print('\n\n')

    print('Training Accuracy: ', accuracy_score(ytrain, yhattrain ))
    print('Testing Accuracy: ', accuracy_score(ytest, yhattest))
    print('\n\n')

    print ('Training F1 Score: ', f1_score(ytrain, yhattrain))
    print ('Testing F1 Score: ', f1_score(ytest, yhattest))
    print ('\n\n')

    
    plot_confusion_matrix(confusion_matrix(ytest, yhattest), classes=class_names)
    print ('\n\n')
    plot_confusion_matrix(confusion_matrix(ytest, yhattest), classes=class_names, normalize=True)
print('evaluate model function created')

def get_authors(list_name=None, df=None, column='author'):
    
    for index, row in df.iterrows():
        list_name.append(row[column])
    return set(list_name)
print('get authors function created')

def get_author_posts(author_list, stop_words, stem_abs, db_collection):
    
    for author in author_list:
        if check_exists(author)==False:
            pass
        else:


            subreddit_dict= {}
            
            for submission in reddit.redditor(author).submissions.new(limit=25):
                if submission.selftext == '':
                    pass
                elif len([i for i in (text_cleaner(submission.selftext, stop_words))])<10:
                    pass

                else:

                    subreddit_dict["created_utc"] = submission.created_utc
                    subreddit_dict["created_at"] = get_date(submission)
                    subreddit_dict["subreddit"] = submission.subreddit.display_name
                    subreddit_dict["author"] = submission.author.name
                    subreddit_dict["_id"] = submission.id
                    subreddit_dict["selftext"] = submission.title + ' ' + submission.selftext
                    subreddit_dict["cleaned_text"] = text_cleaner(submission.selftext, stop_words)
                    subreddit_dict['percent_abs_words'] = round(get_perc_abs(clean_text(submission.selftext, stop_words), stem_abs),2)
                    subreddit_dict["sentiment"] = round(TextBlob(text_cleaner(submission.selftext, stop_words)).sentiment.polarity, 2)
                    subreddit_dict["subjectivity"] = round(TextBlob(text_cleaner(submission.selftext, stop_words)).sentiment.subjectivity,2)
                    subreddit_dict["post_length"] = len(submission.selftext)
                    subreddit_dict["hour_posted"] = get_date(submission).hour
                    try:
                        db_collection.insert_one(subreddit_dict)
                    except Exception as e:
                        print("Could not insert text")
                        print("-"*20)
                    subreddit_dict = {}
    print('all finished')
print('get author posts function created')

def check_exists(username):
    
    exists = True
    try:
        reddit.redditor(username).fullname
    except prawcore.exceptions.NotFound:
        exists = False
    return exists
print('check exists function created')

def date(created):
    d= dt.datetime.fromtimestamp(created)
    return d.astimezone(get_localzone())
print('date function created')

