import multiprocessing
import re, sys, os
import string
import time
import emoji
import nltk
from nltk import word_tokenize
import pandas as pd
import matplotlib
from collections import Counter, defaultdict
from turkish.deasciifier import Deasciifier
from dask import dataframe as dd
import multiprocessing
import cloudpickle
import numpy as np
import nltk
nltk.download('punkt')
class preProcessing():

    def process(self, folder,lemmatize: bool, deasciify: bool, asciify: bool):
        stopwords, df = self.read_all(folder)
        print("grouping messages...")
        df = self.groupby_messages(df, 7)
        if lemmatize:
            print("lemmatizing...")
            df['msg'] = dd.from_pandas(df['msg'], npartitions=4 * multiprocessing.cpu_count()) \
                .map_partitions(lambda df: df.apply(lambda x: self.lemmatize(x, deasciify))) \
                .compute(scheduler="processes")

        print("tokenizing...")
        df['msg'] = dd.from_pandas(df['msg'], npartitions=4 * multiprocessing.cpu_count()) \
            .map_partitions(lambda df: df.apply(lambda x: self.custom_tokenizer(x, False, True, stopwords))) \
            .compute(scheduler="processes")
        print("removing empty tokens...")
        df = df[df['msg'].map(lambda d: len(d)) > 0]
        print("stemming...")
        return df

    def read_all(self, folder):
        files = os.listdir(folder)
        all = []
        f = open('/content/stopwords', 'r')
        stopwords = f.readlines()
        f.close()
        print(files)
        for file in files:
            if file.endswith('.txt'):
                history = self.read_history(file)
                print("reading " + file)
                all.append(history)
        for idx, line in enumerate(stopwords):
            stopwords[idx] = stopwords[idx].strip('\n')
            # deasciify the string
        stopwords = self.stem(stopwords,False, True)

        history = pd.concat(all).reset_index()
        # Media messages appear as <Media omitted>, so I delete them
        history_clean = history[history['msg'] != '<Medya dahil edilmedi>']
        history_clean = history_clean[history_clean['msg'] != 'Cevapsız sesli arama']
        history_clean = history_clean[history_clean['msg'] != 'Cevapsız görüntülü grup araması']
        history_clean = history_clean[history_clean['msg'] != 'Cevapsız görüntülü arama']
        history_clean = history_clean[history_clean['msg'] != 'Bu mesaj silindi']

        return stopwords, history_clean

    def lemmatize(self, text: str, deasciify : bool) -> str:
        result = ''
        lemmatizer = MorphAnalyzer()
        lemmatization = lemmatizer.lemmatize_text(text)
        for sentence, lemmas in lemmatization:
            for word, lemma in lemmas:
                if len(lemma) > 1:
                    result = result + word + ' '
                else:
                    result = result + lemma[0] + ' '
        return result.rstrip()

    def read_history(self, file):
        f = open('/content/{}'.format(file), 'r')
        # Every text message has the same format: date - sender: message.
        messages = re.findall('(\d+.\d+.\d+ \d+\d+:\d+\d+) - (.*?): (.*)', f.read())
        f.close()

        # Convert list to a dataframe and name the columns
        history = pd.DataFrame(messages, columns=['date', 'name', 'msg'])
        history['date'] = pd.to_datetime(history['date'])
        history['date1'] = history['date'].apply(lambda x: x.date())
        history['conv_name'] = file.split(' ')[0]

        return history

    def groupby_messages(self, data, n):
        # Sort messages by conversation and time sent
        new_data = data.sort_values(by=['conv_name', 'date'])

        # Group messages in groups of n messages, sent by the same person on the same day
        new_data['group'] = new_data.groupby(['conv_name', 'date']).cumcount()
        new_data['group'] = new_data['group'].apply(lambda x: np.floor(x / float(n)))
        new_data['msg'] = new_data['msg'].apply(lambda x: ' ' + x)
        new_data = new_data.groupby(['conv_name', 'date', 'group'])['msg'].sum().reset_index()

        return new_data

    def custom_tokenizer(self, text, deasciify, asciify, stopwords):
        # remove punctuation
        remove_punct = str.maketrans('', '', string.punctuation)
        text = text.translate(remove_punct)

        # remove digits and convert to lower case
        remove_digits = str.maketrans('', '', string.digits)
        text = text.lower().translate(remove_digits)
        # remove emojis
        emojis = ''.join(e for e in emoji.UNICODE_EMOJI)
        remove_emojis = str.maketrans('', '', emojis)
        text = text.translate(remove_emojis)

        # remove duplicated letters
        #text = re.sub(r'([a-z])\1+', r'\1', text)

        text = "" if "http" in text else text

        text = re.sub(r'^(http)', '', text)

        # combine 'jaja' expresions (this is how Argentinians laugh)
        text = re.sub(r'(ha)[ha]+', '', text)

        # deasciify sentences
        # deasciifier = Deasciifier(text)
        # text = deasciifier.convert_to_turkish()

        # tokenize
        tokens = word_tokenize(text)

        # remove stop words
        tokens_stop = [y for y in tokens if y not in stopwords]

        tokens_stop = self.stem(tokens_stop, deasciify, asciify)

        return tokens_stop

    def stem(self, tokens: list, deasciify, asciify):
        #stemming
        #print("stems", tokens)
        result = []
        for word in tokens:
            if len(word) < 3:
                continue
            if "http" in word:
                continue
            # deasciify the string
            if deasciify == True:
                deasciifier = Deasciifier(word)
                word = deasciifier.convert_to_turkish()
            if asciify:
                asciify_string = str.maketrans("çğıöşü", "cgiosu")
                word = word.translate(asciify_string)
            word = word if len(word)<7 else word[:6]
            result.append(word)
        return result

# words = defaultdict(int)
# series = history_clean['msg'][:50]
# names = history_clean['name'][:50]
# for idx, sentence in enumerate(series):
#     # print(sentence, idx)
#     #name = names[idx]
#     sentence = sentence.split(' ')
#     for word in sentence:
#         words[word] += 1
# sorted_words = sorted(words.items(), key=lambda kv: kv[1])
#
# print(sorted_words[-50:])

    # obj = detector.TurkishNLP()
    # # obj.download()
    # obj.create_word_set()
    # def correct_words(text):
    #   text = obj.list_words(text)
    #   text = obj.auto_correct(text)
    #   return text