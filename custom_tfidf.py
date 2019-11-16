import numpy as np
import pandas as pd
import spacy


def build_vocab(doc_series, min_df=1):
    '''Builds a vocabulary from a series of spacy doc objects'''

    # Create flat list with all words without stopwords and punctuation
    word_ls = []
    for i in range(len(doc_series)):
        doc = doc_series.iloc[i]
        words = [token.lower_ for token in doc if not token.is_stop and token.pos_ != "PUNCT"]
        word_ls += words

    # Create word frequency dictionary
    word_freqs = {}
    for w in word_ls:
        if w not in word_freqs.keys():
            word_freqs[w] = 1
        else:
            word_freqs[w] += 1

    # Filter word frequency dictionary: only include words with frequency >= min_df
    word_freqs = {k: v for k, v in word_freqs.items() if v >= min_df}

    # Get and sort dictionary keys
    vocab = np.sort(list(word_freqs.keys()))

    return vocab


def build_tf(doc_series, vocabulary):
    '''Builds a term frequency matrix (boolean: 1 if term appears, 0 otherwise)'''

    # Initialize tf-idf matrix with zeros
    tf_matrix = np.zeros([len(doc_series), len(vocabulary)])

    for d in range(len(doc_series)):
        doc = doc_series.iloc[d]

        # Filter words out that are not in dictionary
        lemmas = [token.lower_ for token in doc if
                  not token.is_stop and
                  token.pos_ != "PUNCT" and
                  token.lower_ in vocabulary]

        # tf matrix
        for t in range(len(vocabulary)):
            if vocabulary[t] in lemmas:
                tf_matrix[d,t] = 1

    return tf_matrix


def build_tfidf(tf_matrix):
    '''Builds a tf-idf matrix from a tf matrix'''

    N = tf_matrix.shape[0]
    n_t = np.sum(tf_matrix, axis=0)
    idf_t = np.log(N/n_t)

    tfidf_matrix = np.copy(tf_matrix)

    for t in range(tf_matrix.shape[1]):
        tfidf_matrix[:,t] = tf_matrix[:,t] * idf_t[t]

    return tfidf_matrix


def build_tfidf_complete(doc_series, min_df=1):
    '''Builds a vocabulary from a series of spacy doc objects'''

    vocab = build_vocab(doc_series, min_df=min_df)
    tf = build_tf(doc_series, vocab)
    tfidf = build_tfidf(tf)

    return tf
