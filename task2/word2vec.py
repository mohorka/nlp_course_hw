import re
import sys
from typing import List

import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer

from utils.pymorphy_fix import pymorphy2_311_hotfix

if sys.version_info.minor >= 11:
    pymorphy2_311_hotfix()

nltk.download("stopwords")
patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-«»]+"
stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def preprocess(article: str) -> List[str]:
    """Lemmatize and remove stopwords from given list of russian strings.

    Args:
        data (List[str]): List of russian texts.
        output_path (str, optional): Path to save preprocessed text.
            If it is not set, preprocessed text won't be saved to file.
            Defaults to "".

    Returns:
        List[str]: List of preprocessed text.
    """
    article = article.lower()
    words = re.findall(r"\b\w+\b", article)
    cleaned_words = [word for word in words if word not in stopwords_ru]
    tokens = [morph.normal_forms(word)[0] for word in cleaned_words]
    remove_shorts = [token for token in tokens if len(token) > 2]
    remove_digits = [token for token in remove_shorts if not token.isdigit()]
    return remove_digits


def get_word2vec(tokenized: List[List[str]], output_model_path: str = ""):
    """Train word2vec model.

    Args:
        tokenized (List[List[str]]): Train data.
        output_model_path (str, optional): Path to save train model if it is needed.
            Defaults to "".

    Returns:
        model: Trained w2v model.
    """
    model = Word2Vec(tokenized)
    if output_model_path != "":
        model.save(output_model_path)
    return model


def load_trained_model(model_path: str):
    """Load trained w2v model.

    Args:
        model_path (str): Path to trained model.

    Returns:
        model: Loaded w2v model.
    """
    return Word2Vec.load(model_path)
