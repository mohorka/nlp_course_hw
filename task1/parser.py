import argparse
import logging
from datetime import date
from typing import List

from task1.extractors.extractor import PersonParser
from utils.files_utils import read_data, checkoutput_exists


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="news.txt",
        help="File with texts to parse. Default: %(default)s",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="parsed.txt",
        help="File to write output. Default: %(default)s",
    )
    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    checkoutput_exists(args.output)
    df = read_data(args.input_file)
    content: List[str] = df.content.to_list()
    with open(args.output, "w") as f:
        for text in content:
            for match in PersonParser.findall(text):
                birth_date, location = None, None
                if match.fact.birth is not None:
                    day = match.fact.birth.day
                    month = match.fact.birth.month
                    year = match.fact.birth.year
                    birth_date = date(year, month, day)
                if match.fact.location is not None:
                    location = match.fact.location.city
                parsed = f"{match.fact.name.first}\t{match.fact.name.last}\t{birth_date}\t{location}\n"
                f.write(parsed)


if __name__ == "__main__":
    main()
