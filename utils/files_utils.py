from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def checkoutput_exists(path_to_output: str) -> None:
    """Check if file exists. If it is, raise exception.

    Args:
        path_to_output (str): path to probably existing file.

    Raises:
        RuntimeError: Raise error to prevent execution.
    """
    output_file = Path(path_to_output)
    if output_file.is_file():
        raise RuntimeError(
            f"File {path_to_output} already exists! "
            "It's forbidden to write to existing file, choose another one."
        )


def read_data(path_to_data: str) -> pd.DataFrame:
    """Read data from .txt file as pandas DataFrame.
    Remove null values if its exists.

    Args:
        path_to_data (str): path to file with data.

    Returns:
        pd.DataFrame: Dataframe with removed null values.
    """
    df = pd.read_csv(path_to_data, sep="\t", header=None)
    df.columns = ["topic", "header", "content"]
    df.dropna(inplace=True)
    return df


def write_output(content: List[Dict[str, Optional[str]]], path_to_output: str) -> None:
    """Write result to .csv file.

    Args:
        content (List[Dict[str, Optional[str]]]): list with parsed objects.
        path_to_output (str): path to output file.
    """
    checkoutput_exists(path_to_output)
    df = pd.DataFrame.from_dict(content)
    df.to_csv(path_to_output)
