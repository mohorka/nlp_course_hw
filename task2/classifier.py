import logging
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from task2.word2vec import load_trained_model


def prepare_train_test(
    preprocessed: List[List[str]],
    labels: List[int],
    word2vec_path: str,
    test_size: float = 0.3,
    shuffle: bool = True,
):
    """Prepate train-test data with vectorizing via word2vec and simple mean.

    Args:
        preprocessed (List[List[str]]): List of preprocessed (tokenized) data.
        labels (List[int]): List of labels ids.
        word2vec_path (str): Path to word2vec model.
        test_size (float, optional): Represent the proportion of the dataset
            to include in the test split . Defaults to 0.3.
        shuffle (bool, optional): Whether or not to shuffle the data before splitting.
            Defaults to True.

    Returns:
        Tuple[list, list, list, list]: Data splitted by train and test.
    """
    w2v_model = load_trained_model(word2vec_path)
    vectorized_content = []
    for article in preprocessed:
        article_vector = []
        for word in article:
            try:
                vector = w2v_model.wv[word]
                article_vector.append(vector)
            except KeyError:
                logging.warning(f"{word} was not found by word2vec!")
        article_vector = np.mean(article_vector, axis=0)
        vectorized_content.append(article_vector)

    X_train, X_test, y_train, y_test = train_test_split(
        vectorized_content, labels, test_size=test_size, shuffle=shuffle
    )
    return X_train, X_test, y_train, y_test


def _get_labels_mapping(labels: List[str]) -> Dict[str, int]:
    unique = set(labels)
    mapping = {}
    for idx, label in enumerate(unique):
        mapping[label] = idx

    return mapping


def encode_labels(labels: List[str]) -> List[int]:
    """Encode categorical labels as int.

    Args:
        labels (List[str]): List of categorical labels.

    Returns:
        List[int]: List of labels ids.
    """
    mapping = _get_labels_mapping(labels)
    encoded_labels = [mapping[label] for label in labels]
    return encoded_labels


def train(X_train, y_train):
    """Train logistic regression for multiclass classification. Use L2 penalty.

    Args:
        X_train (_type_): Train data.
        y_train (_type_): Train labels ids.

    Returns:
       model: Trained model.
    """
    model = LogisticRegression(
        solver="liblinear", penalty="l2", C=10, random_state=2024
    )
    model.fit(X=X_train, y=y_train)
    return model
