__author__ = "Christoph Schauer"
__date__ = "2019-06-16"
__version__ = "0.1"


import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nlp = spacy.load("en_core_web_md")


# 0. DATA PREPARATION

def show_random_texts(dataframe, sample_size):
    sample = np.random.randint(low=0, high=len(dataframe), size=sample_size)
    return dataframe["text"].iloc[sample]


# 0. CREATE NLP OBJECTS

def gen_list_of_records(dataframe):
    '''Helper function: Converts a dataframe to a list of records.'''

    import pandas as pd

    # Convert datetime series to datetime
    dataframe["datetime"] = pd.to_datetime(dataframe["datetime"])

    text = dataframe["text"].tolist()
    context = dataframe[["id", "datetime", "location"]].to_dict("records")
    data_ls = list(zip(text, context))
    return data_ls


def gen_nlp(dataframe, company_list):
    '''Converts a dataframe to a list of records, a series of strings into a
    series of spacy doc objects, and properly captures twitter tags and hashtags and
    overwrites their POS as nouns. '''

    # Load spacy language model
    # nlp = spacy.load("en_core_web_md")

    # Rule-based matcher using for twitter tags and hashtags
    matcher = spacy.matcher.Matcher(nlp.vocab)
    matcher.add("TWITTERTAG", None, [{"TEXT": {"REGEX": "^@[\w]+$"}}])
    matcher.add("HASHTAG", None, [{"ORTH": "#"}, {"IS_ASCII": True}])

    # Set extension for token: True/False is token is in company_list
    company_tag_getter = lambda token: token.lower_ in company_list
    spacy.tokens.Token.set_extension("is_company_tag", getter=company_tag_getter)

    data_ls = gen_list_of_records(dataframe)

    record_ls = []

    # Loop over all rows in data_ls
    for doc, context in nlp.pipe(data_ls, as_tuples=True, n_threads=-1,
                                 batch_size=64, disable=["parser", "ner"]):

        # Matches for twitter tags and hashtags
        matches = matcher(doc)
        spans = []
        for match_id, start, end in matches:
            span = doc[start:end]
            spans.append(span)

        # Retokenize to combine hashtags into single tokens
        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(span)

        # Convert POS to nouns for all twitter tags and hashtags
        for token in doc:
            if token.lower_ in [span.lower_ for span in spans]:
                token.pos = 92
                token.pos_ = "NOUN"

        # Extract and add company tags
        company_tags = list(set([token.lower_ for token in doc if token._.is_company_tag]))

        record_ls.append((context["id"], context["datetime"], context["location"], company_tags, doc))

    df = pd.DataFrame.from_records(record_ls, columns=["id", "datetime", "location", "company_tags", "doc"])

    return df


# 1. FACETS

def filter_by_company(dataframe, company):
    '''Filters a dataframe by company'''

    df = dataframe.loc[dataframe["company_tags"].apply(lambda company_tags: company in company_tags)]
    return df


def plot_company_count(dataframe, company_list):
    '''Plots company tag counts'''

    count_ls = [len(filter_by_company(dataframe, c)) for c in company_list]
    s = pd.Series(count_ls, index=company_list).sort_values(ascending=False)
    sns.barplot(x=s, y=s.index, color=sns.color_palette()[0])
    plt.title("Texts per company")
    plt.show()


# 2. ANALYZE TRENDS AND ANOMALIES

def count_by_period(dataframe, period):
    '''Helper function: Returns text counts by period'''

    ts = dataframe["datetime"]
    ts.index = ts

    if period == "date":
        ts = ts.groupby([ts.index.date]).count()
        r = pd.date_range(ts.index.min(), ts.index.max())

    elif period == "week":
        ts = ts.groupby([ts.index.week]).count()
        r = range(ts.index.min(), ts.index.max()+1)

    elif period == "hour":
        ts = ts.groupby([ts.index.hour]).count()
        r = range(0, 24)

    elif period == "weekday":
        ts = ts.groupby([ts.index.weekday]).count()
        r = range(0, 7)

    elif period == "datehour":
        ts = ts.groupby([ts.index.date, ts.index.hour]).count()
        x = ts.reset_index(drop=False, name="count")
        x.columns = ["date", "time", "count"]
        x.index = pd.to_datetime(x["date"]) + pd.to_timedelta(x['time'], unit='h')
        ts = x["count"]
        r = pd.date_range(ts.index.date.min(), ts.index.date.max(), freq="H")

    ts = ts.reindex(r, fill_value=0)
    return ts


def plot_trends(dataframe, period):
    '''Visualizes text count over time. Available periods:
    date, week, hour, weekday, date*hour'''

    ts = count_by_period(dataframe, period)
    plt.figure(figsize=(8,4))

    if period == "datehour":
        sns.lineplot(x=ts.index, y=ts, color=sns.color_palette()[0])
    else:
        sns.barplot(x=ts.index, y=ts, color=sns.color_palette()[0])

    plt.xlabel(period)
    plt.ylabel("count")
    plt.title("Number of texts by " + period)
    plt.show()


def heatmap_company(dataframe, company_list, period, figsize):
    '''Function to plot a text frequecy: company by period heatmap'''

    ls = []
    for c in company_list:
        df = filter_by_company(dataframe, c)
        ts = count_by_period(df, period)
        ls.append(ts)
    m = np.array([ls]).squeeze()
    m = pd.DataFrame(m)
    m.index = company_list

    plt.figure(figsize=figsize)

    sns.heatmap(m)
    plt.ylabel("company")
    plt.xlabel(period)
    plt.title("text count heatmap")
    plt.show()


def heatmap_datehour(dataframe, figsize):
    '''Function to plot a text frequecy: date by hour heatmap'''

    ts = count_by_period(dataframe, "datehour")
    ts = pd.DataFrame(ts)
    ts["x"] = ts.index.hour
    ts["y"] = ts.index.date
    m = pd.pivot_table(ts, index="y", columns="x", values="count",
                   aggfunc=np.sum)

    plt.figure(figsize=figsize)

    sns.heatmap(m)
    plt.ylabel("date")
    plt.xlabel("hour")
    plt.title("text count heatmap")
    plt.show()


# UNUSUAL WORDS

def get_topk_terms(dataframe, vocabulary, tfidf_matrix, topk):
    '''Returns the top k most prominent terms'''

    m = tfidf_matrix[dataframe.index]
    tfidf_sums = np.sum(m ,axis=0)
    vocab_df = pd.DataFrame({"vocab":vocabulary, "tfidf":tfidf_sums}).sort_values(by="tfidf", ascending=False)

    return vocab_df.head(topk)

def draw_wordcloud(dataframe, max_words, tfidf_matrix, vocabulary):
    '''Creates a wordcloud from the most prominent terms'''

    df = get_topk_terms(dataframe, vocabulary, tfidf_matrix, max_words)
    freqs = dict(zip(df["vocab"], df["tfidf"]))

    # Define wordcloud mask
    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
    mask = 255 * mask.astype(int)

    # Draw wordcloud
    wc = WordCloud(background_color="white", mask=mask, max_words=max_words)
    wc.generate_from_frequencies(freqs)
    plt.figure(figsize=(8,8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# KEYWORD FILTER

def contains_keyword(doc, keyword_list):
    '''Helper function: Returns True if doc contains a keyword from a list, else False'''
    x = False
    for token in doc:
        if token.lower_ in keyword_list:
            x = True
            break
    return x

def filter_by_keyword(dataframe, keyword_list):
    '''Returns all texts that include keywords from a list'''
    s = dataframe["doc"]
    s = s[s.apply(contains_keyword, args=(keyword_list,))]
    s = s.apply(lambda doc: doc.text)
    return s
