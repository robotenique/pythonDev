import json
import toolz as fp
import pandas as pd
from typing import List, Dict
from pprint import pprint
from datetime import datetime



def process_cards(card_list):
    keys_to_keep = [
        "name",
        "dateLastActivity",
        "desc",
        "id",
        "idLabels",
        "idList",
        "labels",
        "pos",
    ]
    return list(map(lambda c: fp.keyfilter(lambda k: k in keys_to_keep, c), card_list))


def process_lists(lists_list):
    keys_to_keep = ["id", "name", "pos"]
    return list(map(lambda c: fp.keyfilter(lambda k: k in keys_to_keep, c), lists_list))


def process_labels(labels_list):
    keys_to_keep = ["color", "id", "name"]
    return list(
        map(lambda c: fp.keyfilter(lambda k: k in keys_to_keep, c), labels_list)
    )


def process_actions(actions_list):
    keys_to_keep = ["id", "data", "date", "type"]
    return list(
        map(lambda c: fp.keyfilter(lambda k: k in keys_to_keep, c), actions_list)
    )


def flatten_json(entity_list):
    return list(
        map(
            lambda a: pd.json_normalize(a, sep="_").to_dict(orient="records")[0],
            entity_list,
        )
    )


def main():
    # set to True to regenerate the data files again from the raw json
    REGENERATE_FILES = True
    # get current time and format as a string
    curr_date = datetime.now().strftime("%Y_%m_%d")
    # json_file: str = "/home/robotenique/pythonDev/util/trello_dump_audiovisual_2022_11.json"
    json_file: str = "/home/robotenique/docs/pythonDev/util/trello-analysis/trello_audiovisual_data_backup_2023_08_20.json"
    with open(json_file, encoding="utf-8") as infile:
        trello_json: Dict = json.load(infile)

    labels = trello_json["labelNames"]
    last_update: str = trello_json["dateLastActivity"]

    detailed_labels: List = process_labels(trello_json["labels"])
    detailed_lists: List = process_lists(trello_json["lists"])
    detailed_actions: List = flatten_json(process_actions(trello_json["actions"]))
    detailed_cards: List = flatten_json(process_cards(trello_json["cards"]))

    if REGENERATE_FILES:
        pd.DataFrame(detailed_labels).to_pickle(f"{curr_date}_labels_df.pkl")
        pd.DataFrame(detailed_lists).to_pickle(f"{curr_date}_lists_df.pkl")
        pd.DataFrame(detailed_actions).to_pickle(f"{curr_date}_actions_df.pkl")
        pd.DataFrame(detailed_cards).to_pickle(f"{curr_date}_cards_df.pkl")

if __name__ == "__main__":
    main()
