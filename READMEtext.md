
# Text Summarization

Have you ever summarized a lengthy document into a short paragraph? How long did you take? Manually generating a summary can be time consuming and tedious. Automatic text summarization promises to overcome such difficulties and allow you to generate the key ideas in a piece of writing easily.

Text summarization is the technique for generating a concise and precise summary of voluminous texts while focusing on the sections that convey useful information, and without losing the overall meaning. Automatic text summarization aims to transform lengthy documents into shortened versions, something which could be difficult and costly to undertake if done manually. Machine learning algorithms can be trained to comprehend documents and identify the sections that convey important facts and information before producing the required summarized texts.

Broadly, there are two approaches to summarizing texts in NLP: extraction and abstraction.

Extractive Summarization: In this process, we focus on the vital information from the input sentence and extract that specific sentence to generate a summary. There is no generation of new sentences for summary, they are exactly the same that is present in the original group of input sentences.

In abstraction-based summarization, advanced deep learning techniques are applied to paraphrase and shorten the original document, just like humans do. Think of it as a pen—which produces novel sentences that may not be part of the source document.

Now, we will be using different different models corresponding to deep learning to build a solution to our problem statement. The code is basically python. We have imported various dictionaries, took reference from various libraries and did the project.


## Description

- Trained and tested over a detailed dataset using Kaggle
- various machine learning approaches towards the model and ran various epochs to make it better.


## Usage and Installation

```

import numpy as np
import pandas as pd
import pickle
from statistics import mode

import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import backend as K 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

import pickle as pickle

import pathlib as pathlib

x_train,x_test,y_train,y_test=train_test_split(input_texts,target_texts,test_size=0.2,random_state=0) 
```


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

We got an accuracy of 85.25 % in this model. We ran various epochs and used efficiet data cleansing techniques to get to this.

## Used By

The project is used by a lot of social media companies to analyse their market.


