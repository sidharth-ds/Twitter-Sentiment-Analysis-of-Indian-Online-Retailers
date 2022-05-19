from flask import Flask,render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
from os.path import join, dirname, realpath
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# load the model from disk:
filename = 'Sentiment_analysis.pkl'
model = pickle.load(open(filename, 'rb'))    # read
filename = 'transformer.pkl'
vectorizer = pickle.load(open(filename, 'rb'))

# run the flask app
app = Flask(__name__)
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# home page:
@app.route('/')
def home():
    return render_template('home.html')

# result page:
@app.route('/result',methods=['POST'])
def predict():
    if request.method == 'POST':            # to connect html
        uploaded_file = request.files['myfile']
        if uploaded_file.filename != '':
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)  # path specify
            # set the file path
            uploaded_file.save(file_path)

            message = pd.read_csv('static/files/test.csv')   # read

            message = message.iloc[:,1]
            message = [cleantweet(i) for i in message]
            message = [remove_emojis(i) for i in message]
            message = [i.lower() for i in message]
            message = [i.strip() for i in message]
            message = [lemmatize_sentence(i) for i in message]

            message = vectorizer.transform(message)    # vectorization
            my_prediction = model.predict(message)     # predict

            return render_template('result.html',value=round(my_prediction.mean(),3)) # return the mean of predictions

# required functions:
def cleantweet(tweet):
    tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)     # to remove @
    tweet = re.sub(r'#', '', tweet)                # to remove hashtags
    tweet = re.sub(r'RT[\s]', '', tweet)           # to remove retweets
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)   # to remove hyperlinks
    tweet = re.sub(r'[^\w\s]', '', tweet)          # to remove punctuations
    tweet = re.sub(r'\n', '', tweet)               # to remove next line
    tweet = re.sub(r'_', '', tweet)                # to remove underscore
    tweet = re.sub(" \d+", "", tweet)              # to remove numericals
    return tweet

def remove_emojis(tweet):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))

    # tuple of (token, wordnet_tag)     # convert detailed POS into shallow POS
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

if __name__ == '__main__':
	app.run(debug=True)


