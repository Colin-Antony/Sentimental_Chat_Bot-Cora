import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# package that contain 7500 english words that are useful in analyzing the sentiment of a sentence.


def analyse_sentiment(input_string, score_array):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(input_string)
    print(sentiment_scores)
    score_array[0] += sentiment_scores['compound']
    score_array[1] += sentiment_scores['pos']
    score_array[2] += sentiment_scores['neg']
    score_array[3] += sentiment_scores['neu']
    if input_string != "Summary":
        score_array[4] += 1
    return score_array      # comp,pos,neg,neu

def analyse_senti(str):
    sid=SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(str)
    return scores['compound']

def Sentence_segmenter(input_string):
    input_string = nltk.sent_tokenize(input_string)
    print(input_string)
    return input_string


a="I hate elephants"
print(analyse_senti(a))

# scores = [0, 0, 0, 0,0]
# while(a!=["Summary"]):
#     a = input("Enter your sentence\n")
#     a = Sentence_segmenter(a)
#     print(a)
#     for i in a:
#         sentiment_score = analyse_sentiment(i, scores)
#         print(sentiment_score)  # rounds of decimals to 4 places
#
# if sentiment_score[0]/sentiment_score[4]>0.5:
#     print("You are positive")
# else:
#     print("You are neggy man")

