import math
import logging
import json
import os


def load_config(path):
    """
    Load configuration file with all the needed parameters
    """
    with open(path, 'r') as conf_file:
        conf = json.load(conf_file)
    return conf


def get_logger(name):
    """
    Add a StreamHandler to a logger if still not added and
    return the logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 1  # propagate to parent
        console = logging.StreamHandler()
        logger.addHandler(console)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
    return logger


utils_log = get_logger(__name__)
utils_log.setLevel(logging.INFO)


def csv_maker(arg_input, csv_file):
    """
    Produce the csv location and stock name string
    by looking at the folder name.
    :param arg_input: inpyt argoment
    :param csv_file: csv string
    :return: csv to be read by pandas and sstock name
    """
    path = str(arg_input)
    csv = os.path.join(path, csv_file)
    csv_folder_string = str(csv_file.strip(".csv"))
    return csv, csv_folder_string


def create_folder(path, string):
    """
    Create a folder
    :param path: starting path
    :param string: folder name
    :return: path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    output_dir = os.path.join(path, string)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def store_csv(df, path, csv_name):
    """
    Store a df in csv file.
    :param df: pandas dataframe
    :param path: path to output
    :param csv_name: csv name
    :return: None
    """

    csv_loc = os.path.join(path, csv_name + ".csv")
    df.name = csv_name
    df.to_csv(csv_loc)


def asc_sort_tuple(tup):
    """
    It loop over the tuple and sort it
    ascending by looking at the second
    item.
    :param tup: tuple list
    :return: sorted tuple list
    """
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if tup[j][1] > tup[j + 1][1]:
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


def disc_sort_tuple(tup):
    """
    It loop over the tuple and sort it
    descending by looking at the second
    item.
    :param tup: tuple list
    :return: sorted tuple list
    """
    lst = len(tup)
    for i in range(0, lst):
        for j in range(0, lst - i - 1):
            if tup[j][1] < tup[j + 1][1]:
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup


def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)