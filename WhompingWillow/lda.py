# Created by White Wolf
# Date: 8/15/20

import pyLDAvis.gensim
from gensim.models.ldamodel import LdaModel


def build_lda(dictionary, BoW) -> LdaModel:
    """
    Build the LDA model.
    :param dictionary:
    :param BoW:
    :return ldaModel: Gensim LDA Model
    """
    return LdaModel(
        corpus=BoW, id2word=dictionary, num_topics=10, random_state=100,
        update_every=1, chunksize=500000, passes=1, alpha='auto', per_word_topics=True)


def display_lda(lda, dictionary, BoW) -> None:
    """
    Display LDA with PyLDAVis
    :param lda:
    :param dictionary:
    :param BoW:
    :return nothing:
    """
    lda_prepare = pyLDAvis.gensim.prepare(lda, BoW, dictionary)
    pyLDAvis.show(lda_prepare)  # ToDo change this so it runs outside of the program so the program continues to run...
