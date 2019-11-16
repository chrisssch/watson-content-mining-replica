# IBM Watson Content Mining Replica

Author: Christoph Schauer

Created: 2019/06/16 / Uploaded: 2019/11/16

Version: 0.1


## Introduction

In this project I'm replicating the key functionalities of [IBM's Watson Explorer Deep Analytics Edition](https://www.ibm.com/nl-en/products/watson-explorer) Content Mining (as shown [here](https://www.youtube.com/watch?v=B9SMcP1w3_o)) in Python on a basic level. For the time being this is primarily a little exercise in NLP with spacy and programming.

The dataset used is the [Twitter US Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment) from Kaggle.
Twitter data is not really ideal for this purpose, but this dataset is small, easy to handle, and comes with date fields, which is required here.

This project uses two scripts, content_mining.py and custom_tfidf.py. content_mining.py includes most of the code, custom_tfidf includes functions that build a (basic) tf-idf matrix from a series of spacy doc objects. How these functions work is shown in the Jupyter notebook in this repository.


## Next Steps

* Convert collection of functions to an object and clean up code
