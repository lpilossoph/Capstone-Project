import requests
import nltk
import json
import time
import pandas as pd
import re
import praw
import datetime as dt
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
from pymongo import errors
from pymongo.errors import InvalidDocument
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from pymongo import MongoClient
from functions import connect_to_reddit, connect_to_mongo, get_stop_words, text_cleaner, submissions_to_mongo, clean_df, get_perc_abs, get_abs_chart, get_post_distribution, wordcloud, clean_text, get_distribution, get_stem_abs

client = connect_to_mongo()
reddit = connect_to_reddit()

reddit_db = client["reddit"]
print(reddit_db.list_collection_names())

abs_word_list = ['absolutely', 'all', 'always', 'complete', 'completely', 'constant', 'constantly', 'definitely', 
                 'entire', 'ever', 'every', 'everyone', 'everything', 'full', 'must', 'never', 'nothing', 
                 'totally', 'whole']

remove_words = ["http", "https", "www", "com", "reddit", "malefashionadvice", "jpg"]
pre_stop_words = list(stopwords.words("english"))

moderators = ['S2S2S2S2S2','Psy-Kosh','SicSemperHumanus','SQLwitch',
            'UnDire','pkbooo','skyqween','MykeeB',
            'circinia','svneko','MuffinMedic','remyschnitzel',
            'vodkalimes','dwade333miami','anxietymods',
            'BotBust','MrZalarox','Pi25','analemmaro', 'abhava-sunya', 
            'sofar1776','Kalium','AutoModerator',
            'Thonyfst','sconleye','citaro','Criminal_Pink','Smilotron',
            'evolsirhc','thegreatone3486','trackday_bro','Greypo',
            'exoendo','Jakeable','MeghanAM','JoyousCacophony','hansjens47',
            'Qu1nlan','english06','samplebitch','optimalg','rhiever',
            'Geographist','frostickle','NonNonHeinous','Vizual-Statistix',
            'sarahbotts','zonination','yelper','mungoflago','ostedog',
            'rsrsrsrs','townie_immigrant','JoeAllan','theReluctantHipster',
            'AutoModerator', 'iimsorryy']

subreddit_list = ['suicidewatch', 'depression', 'anxiety', 
                'malefashionadvice', 'travel', 'basketball']

stem_abs = get_stem_abs(abs_word_list)

reddit_submissions = reddit_db["forums_3"]

stop_words = get_stop_words(remove_words, abs_word_list, pre_stop_words)
print(stop_words)

reddit_df = submissions_to_mongo(subreddit_list, reddit_submissions, reddit, moderators, stop_words)
print(reddit_df.head())



