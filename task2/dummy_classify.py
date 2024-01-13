import argparse
import logging
from typing import List

from sklearn import metrics

from task2.classifier import encode_labels, prepare_train_test, train
from task2.word2vec import get_word2vec, preprocess
from utils.files_utils import read_data


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="news.txt",
        help="Path to file with texts. Default: %(default).",
    )
    parser.add_argument(
        "--trained_w2v",
        type=str,
        help="Path to trained word2vec model.",
    )
    parser.add_argument(
        "--path_to_save_w2v",
        type=str,
        help="Path to save fresh word2vec model.",
    )
    args = parser.parse_args()
    return args


def dummy_classify(
    content: List[str], labels: List[str], path_to_w2v: str, train_w2v: bool
) -> float:
    logging.info("Starting data preparation...")
    tokenized = [preprocess(article) for article in content]
    logging.info("Finished data preparation!")
    if train_w2v:
        logging.info("Starting word2vec training...")
        get_word2vec(tokenized, path_to_w2v)
        logging.info("Finished word2vec training!")

    encoded_labels = encode_labels(labels)

    X_train, X_test, y_train, y_test = prepare_train_test(
        tokenized, encoded_labels, path_to_w2v
    )

    logging.info("Start logistic regression training..")
    lr_model = train(X_train, y_train)
    logging.info("Finished logistic regression training!")
    y_pred = lr_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    train_w2v = False
    path_to_w2v = args.trained_w2v

    if args.path_to_save_w2v is not None:
        logging.info(
            "Attention: you chose to train word2vec model from scratch!"
            f"New model will be saved at {args.path_to_save_w2v}"
        )
        train_w2v = True
        path_to_w2v = args.path_to_save_w2v
    elif args.trained_w2v is None:
        raise ValueError(
            "Path to word2vec model isn't set! "
            "Select either the path where to save the new model "
            "or the path to the existing one."
        )
    data = read_data(args.input)
    labels = data.topic.to_list()
    content = data.content.to_list()

    dummy_accuracy = dummy_classify(
        content=content, labels=labels, path_to_w2v=path_to_w2v, train_w2v=train_w2v
    )
    logging.info(
        f"Accuracy for logistic regression + average word2vec embeddings: {dummy_accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
