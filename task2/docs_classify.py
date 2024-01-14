import argparse
import logging
from typing import List

from sklearn import metrics

from task2.classifier import encode_labels, prepare_train_test_via_d2v, train
from task2.vectorizers import get_doc2vec, preprocess, prepare_tagged_docs
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
        "--trained_d2v",
        type=str,
        help="Path to trained doc2vec model.",
    )
    parser.add_argument(
        "--path_to_save_d2v",
        type=str,
        help="Path to save fresh doc2vec model.",
    )
    args = parser.parse_args()
    return args


def docs_classify(
    content: List[str], labels: List[str], path_to_d2v: str, train_d2v: bool
) -> float:
    logging.info("Starting data preparation...")
    tokenized = [preprocess(article) for article in content]
    logging.info("Finished data preparation!")
    if train_d2v:
        logging.info("Tagging docs for doc2vec training...")
        tagged_docs = list(prepare_tagged_docs(tokenized))
        logging.info("Starting doc2vec training...")
        get_doc2vec(tagged_docs, path_to_d2v)
        logging.info("Finished doc2vec training!")

    encoded_labels = encode_labels(labels)

    X_train, X_test, y_train, y_test = prepare_train_test_via_d2v(
        tokenized, encoded_labels, path_to_d2v
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
    train_d2v = False
    path_to_d2v = args.trained_d2v

    if args.path_to_save_d2v is not None:
        logging.info(
            "Attention: you chose to train doc2vec model from scratch!"
            f"New model will be saved at {args.path_to_save_d2v}"
        )
        train_d2v = True
        path_to_d2v = args.path_to_save_d2v
    elif args.trained_d2v is None:
        raise ValueError(
            "Path to doc2vec model isn't set! "
            "Select either the path where to save the new model "
            "or the path to the existing one."
        )
    data = read_data(args.input)
    labels = data.topic.to_list()
    content = data.content.to_list()

    docs_accuracy = docs_classify(
        content=content, labels=labels, path_to_d2v=path_to_d2v, train_d2v=train_d2v
    )
    logging.info(
        f"Accuracy for logistic regression + doc2vec embeddings: {docs_accuracy:.2f}"
    )


if __name__ == "__main__":
    main()
