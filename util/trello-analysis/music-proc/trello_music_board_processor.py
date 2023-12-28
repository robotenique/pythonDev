import json
import toolz as fp
from operator import itemgetter
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import requests
import yaml

JsonRead = Union[List[Any], Dict[str, Any]]


"""
    trello_music_board_processor.py
    The purpose of this script is to process the JSON data from a Trello board backup;

    It's main purpose is to clean up the data, not visualize it or do analysis on it.

    General steps to generate the exported JSON from Trello:
    1. Go to the board you want to export
    2. Click on the Menu
    3. Click on "Print, export, and share"
    4. Click on "Export as JSON"

    But the process above is not perfect - what can be exported is limited - so to get the complete data (full picture),
    we need to manually query the Trello API to get the actual actions we want.
"""

    
# Data loading

def load_data(json_file_path: str) -> Dict[str, Any]:
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data

def get_api(yaml_file_name="trello_config.yaml"):
    print(f"\t\tTrying to load API info from {yaml_file_name}...")
    with open(yaml_file_name, "r") as file:
        config = yaml.safe_load(file)
    # assert that the yaml contains the correct keys: API_KEY, API_TOKEN
    assert config.get("API_KEY") is not None
    assert config.get("API_TOKEN") is not None
    return config

# Data Cleaning
def remove_empty_values(data: JsonRead) -> JsonRead:
    """
    Recursively removes empty values from a JSON-like data structure.

    Args:
        data (JsonRead): The JSON-like data structure to remove empty values from.

    Returns:
        JsonRead: The data structure with empty values removed.
    """
    if isinstance(data, list):
        return [remove_empty_values(item) for item in data if item]
    elif isinstance(data, dict):
        return {key: remove_empty_values(value) for key, value in data.items() if value}
    else:
        return data

def remove_archived(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes archived cards from the given list of cards.

    Args:
        cards (list): A list of cards.

    Returns:
        list: A new list of cards without the archived cards.
    """
    return [card for card in cards if not card.get("closed")]


# Data Extraction 
@fp.curry
def extract_valid_keys(
    cards: JsonRead, valid_keys: Tuple[str] = ("id", "name", "desc", "idList", "labels", "stars", "pos")
) -> List[Dict[str, Any]]:
    """
    Extracts the specified valid keys from the given list of cards, since not all keys are needed.

    Args:
        cards (list): A list of cards.
        valid_keys (tuple, optional): A tuple of valid keys to extract from the cards.
                    Defaults to ("id", "name", "desc", "idList", "labels", "stars", "pos").

    Returns:
        list: A list of dictionaries containing the extracted valid keys from each card.
    """
    valid_keys = list(valid_keys)

    def valid_map(card):
        cleaned_card = {}
        for key in valid_keys:
            cleaned_card[key] = card.get(key)
        return cleaned_card

    return [valid_map(card) for card in cards]


def extract_stars_from_plugin_data(
    cards: List[Dict[str, Any]], starPluginId: str = "5e40446acc3ae64e435cabc9"
) -> List[Dict[str, Any]]:
    """
    Extracts the stars from the plugin data of each card in the given list of cards.
    Since the 'stars' field is not built-in to trello, we need to extract it from the plugin data.
    The pluginId was determined by looking at the JSON data itself.

    Args:
        cards (list): A list of cards.
        starPluginId (str, optional): The ID of the star plugin. Defaults to "5e40446acc3ae64e435cabc9".

    Returns:
        list: The modified list of cards with the "stars" field updated.
    """
    for card in cards:
        card["stars"] = None
        plugin_data_list = card.get("pluginData", [])
        for plugin_data in plugin_data_list:
            if plugin_data.get("idPlugin") == starPluginId:
                value = plugin_data.get("value")
                if value:
                    value_json = json.loads(value)
                    card["stars"] = value_json.get("stars")
    return cards

# Data Organization
@fp.curry
def add_list_info(cards: List[Dict[str, Any]], list_mapping: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adds list information to each card in the given list of cards.

    Args:
        cards (List[Dict[str, Any]]): The list of cards to update.
        list_mapping (Dict[str, Dict[str, Any]]): A dictionary mapping list IDs to list information.

    Returns:
        List[Dict[str, Any]]: The updated list of cards with added list information.
    """
    for card in cards:
        card["list_info"] = list_mapping.get(card["idList"], {})
    return cards


def organize_cards_by_list(cards: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organizes a list of cards by their respective lists.

    Args:
        cards (List[Dict[str, Any]]): A list of cards.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary where the keys are the list names and the values are lists of cards belonging to each list.
    """
    for card in cards:
        card["listName"] = card["list_info"]["name"]
    grouped_cards = fp.groupby("listName", cards)
    # Sort cards in each group by 'pos'
    sorted_grouped_cards = {list_id: sorted(group, key=itemgetter("pos")) for list_id, group in grouped_cards.items()}
    return sorted_grouped_cards

# Data Processing
def process_list_structure(data: JsonRead) -> Dict[str, Dict[str, Any]]:
    cleaned_list: List[Dict[str, Any]] = fp.pipe(data["lists"], extract_valid_keys(valid_keys=("id", "name", "pos")))
    # this maps the list id to the list info
    list_id_to_info: Dict[str, Dict[str, Any]] = {lst["id"]: lst for lst in cleaned_list}
    return list_id_to_info

def process_card_structure(
    data: JsonRead, list_id_to_info: Dict[str, Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    cleaned_cards: List[Dict[str, Any]] = fp.pipe(
        data["cards"],
        remove_archived,
        extract_stars_from_plugin_data,
        extract_valid_keys,
        add_list_info(list_mapping=list_id_to_info),
        organize_cards_by_list
    )
    return cleaned_cards


def general_trello_music_pipeline(
    json_file_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], JsonRead]:
    # 1. Load the data
    print(f"1. Loading data from {json_file_path}...")
    data: JsonRead = load_data(json_file_path)
    # 2. Remove empty values
    print("2. Removing empty values...")
    cleaned_data: JsonRead = remove_empty_values(data)
    # 3. Process the lists data
    print("3. Processing the lists data...")
    list_id_to_info: Dict[str, Dict[str, Any]] = process_list_structure(cleaned_data)
    # 4. Process the cards data
    print("4. Processing the cards data...")
    cards_with_stars: Dict[str, Dict[str, Any]] = process_card_structure(cleaned_data, list_id_to_info)
    # now, cards_with_stars is organized like the Trello UI itself List -> Card order
    print("Done!")
    return list_id_to_info, cards_with_stars, cleaned_data

# API request
def trello_api_request_actions(
    boardId: str, typeAction: str = "createCard", date_filter: str = "2023-01-01", limit: int = 1000
) -> List[Dict[str, Any]]:
    # We need to manually query the TrelloAPI to get the actual actions
    # since the exported Trello JSON has a cutoff on the actions.
    # https://community.atlassian.com/t5/Trello-questions/My-Board-Activity-does-not-go-past-a-certain-date/qaq-p/1362031
    # https://developer.atlassian.com/cloud/trello/guides/rest-api/nested-resources/#actions-nested-resource
    def format_action_create(action):
        return {
            "id_action": action["id"],
            "type": action["type"],
            "date": action["date"],
            "card_id": action["data"]["card"]["id"],
        }

    url_endpoint = f"https://trello.com/1/boards/{boardId}/actions"
    trello_api_config = get_api()
    query = {
        "key": trello_api_config["API_KEY"],  # API
        "token": trello_api_config["API_TOKEN"],  # API
        "before": date_filter,
        "filter": typeAction,
        "limit": limit,
    }

    response = requests.request("GET", url_endpoint, params=query)
    if response.status_code == 200:
        actions = response.json()
        actions_f = []  # to filter
        for a in actions:
            actions_f.append(format_action_create(a))
        print("\t Data retrieved successfully")
        return actions_f
    print("\t Failed to retrieve data:", response.status_code)

@fp.curry
def build_creation_at_data(cards_df: pd.DataFrame, board_id: str) -> pd.DataFrame:
    date_range = pd.date_range(start="2020-01-01", end="2023-12-31", freq="Y")
    date_range = sorted(date_range.strftime("%Y-%m-%d").tolist())
    mega_data_aggregator = []
    for dt in date_range:
        print(f"\t Retrieving data for <= {dt}")
        curr_res = trello_api_request_actions(
            boardId=board_id,
            typeAction="createCard",
            date_filter=dt,
        )
        print(f"\t Total entries: [{len(curr_res)}]")
        mega_data_aggregator.extend(curr_res)
    
    # remove duplicate entries 
    unique_actions = set([ac["id_action"] for ac in mega_data_aggregator])
    filtered_actions = []
    for action_id in unique_actions:
        for ac in mega_data_aggregator:
            if ac["id_action"] == action_id:
                filtered_actions.append(ac)
                break
    createdAt_actions = pd.DataFrame(filtered_actions).rename(columns={"date": "createdAt", "type": "action_type"})
    # transform to dt then format to string
    createdAt_actions["createdAt"] = pd.to_datetime(createdAt_actions["createdAt"]).dt.strftime("%Y-%m-%d")
    cards_df = cards_df.merge(createdAt_actions, left_on="id", right_on="card_id", how="left")
    return cards_df


# Data Conversion 
def cards_to_dataframe(cards_with_stars: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Convert a dictionary of cards with stars into a pandas DataFrame.

    Args:
        cards_with_stars (Dict[str, List[Dict[str, Any]]]): A dictionary where the key is the list name
            and the value is a list of cards (dictionaries) for that list.

    Returns:
        pd.DataFrame: A DataFrame containing all the cards with additional 'list_name' column.

    """
    all_cards = []
    for list_name, cards in cards_with_stars.items():
        for card in cards:
            card["list_name"] = list_name
            all_cards.append(card)
    df_all_cards = pd.DataFrame(all_cards)
    # We need to handle cases where 'stars' could be None (no rating)
    # We replace None with a string for it to be counted in value_counts()
    df_all_cards["stars"].fillna("No Rating", inplace=True)

    return df_all_cards

if __name__ == "__main__":
    json_file_path = "trello_music_board_data_backup_2023_11_20.json"
    # 1. Load the data
    data: JsonRead = load_data(json_file_path)
    # 2. Remove empty values
    cleaned_data: JsonRead = remove_empty_values(data)
    print(cleaned_data["id"])
    # 3. Process the lists data
    list_id_to_info: Dict[str, Dict[str, Any]] = process_list_structure(cleaned_data)
    # 4. Process the cards data
    cards_with_stars: Dict[str, Dict[str, Any]] = process_card_structure(cleaned_data, list_id_to_info)
    # now, cards_with_stars is organized like the Trello UI itself List -> Card order
    print(cards_with_stars.keys())
