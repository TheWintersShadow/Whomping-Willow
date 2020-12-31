# Created by White Wolf
# Date: 7/25/20

import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer


def prep_data(corpus) -> pd.DataFrame:
    """
    Process the data
    :param corpus: PD of the cleaned data. (Needs to have "data" column with the values for LDA.)
    :return processed_docs: Processed PD
    """

    def preprocess(text) -> list:
        """
        Drop any words that are in the STOPWORDS in Gensim's Stop Word list.
        :param text:
        :return result: list of items that are not in STOPWORDS and in the emails
        """

        def lemm_and_stemm(text) -> WordNetLemmatizer:
            """
            Preform Lemming and Stemming on the messages.
            :param text:
            :return a wnl.lemmatized item:
            """
            wnl = WordNetLemmatizer()
            return wnl.lemmatize(WordNetLemmatizer().lemmatize(text, pos='v'))

        result = []
        for token in simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:
                result.append(lemm_and_stemm(token))
        return result

    processed_docs = corpus['data'].map(preprocess)
    return processed_docs


def build_BoW(processed_docs) -> dict:
    """
    Build the Bog of Words and Gensim Dictionary.
    :param processed_docs: PD of processed information.
    :return BoW, id2word: Bag of Words and Gensim Dictionary
    """
    id2word = corpora.Dictionary(processed_docs, prune_at=1000000)
    id2word.filter_extremes(no_below=20, no_above=0.5)
    BoW = [id2word.doc2bow(t) for t in processed_docs]
    return BoW, id2word
