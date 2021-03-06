import functools
import socket
import requests
import arff

import gc
import logging
import operator
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path

import cloudpickle
import numpy as np
import openml
import pandas as pd
import toolz as fp
from sklearn.model_selection import train_test_split

from fklearn.training.classification import lgbm_classification_learner
from exploration import dataset_analyzer, statistics_dict_to_df
from fklearn.training.pipeline import build_pipeline
from fklearn.training.transformation import label_categorizer
from fklearn.validation.evaluators import (
    auc_evaluator,
    brier_score_evaluator,
    combined_evaluators,
    logloss_evaluator,
)


# ----------------------------- Global Contants ---------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


RND_STATE_SEED = 42
PIPELINE_TYPE = "Binary Classifier"

DATASET_PARAMETERS = dotdict(
    dict(
        min_num_features=3,
        min_num_instances=1000,
        max_num_classes=100,
        max_num_instances=5000000,
        max_nan_percentage=0.2,
        train_percentage=0.7,
    )
)

HYPERPARAMETERS_CONFIG = dotdict(
    dict(
        num_estimators_length=20,  # number of num_estimator values to generate
        lr_length=6,  # number of learning rate values to generate
        max_depth_length=20,  # number of max_depth values to generate,
        default_lgbm_main_params={"num_estimators": 100, "learning_rate": 0.1},
        default_lgbm_extra_params={"seed": RND_STATE_SEED, "nthread": cpu_count(), "verbose": -1},
    )
)

VALID_DATASETS_DID = [
    3,
    31,
    44,
    72,
    73,
    77,
    120,
    121,
    122,
    124,
    126,
    128,
    131,
    132,
    135,
    137,
    139,
    140,
    142,
    143,
    146,
    151,
    152,
    153,
    161,
    162,
    179,
    246,
    251,
    256,
    257,
    258,
    260,
    262,
    264,
    267,
    269,
    273,
    274,
    293,
    310,
    312,
    316,
    350,
    351,
    354,
    357,
    715,
    718,
    720,
    722,
    723,
    725,
    727,
    728,
    734,
    735,
    737,
    740,
    741,
    743,
    751,
    752,
    761,
    772,
    797,
    799,
    803,
    806,
    807,
    813,
    816,
    819,
    821,
    823,
    833,
    837,
    843,
    845,
    846,
    847,
    849,
    866,
    871,
    881,
    897,
    901,
    903,
    904,
    910,
    912,
    913,
    914,
    917,
    923,
    934,
    953,
    958,
    959,
    962,
    966,
    971,
    976,
    977,
    978,
    979,
    980,
    983,
    991,
    995,
    1019,
    1020,
    1021,
    1022,
    1037,
    1038,
    1039,
    1040,
    1042,
    1046,
    1049,
    1050,
    1053,
    1056,
    1067,
    1068,
    1069,
    1116,
    1119,
    1120,
    1128,
    1130,
    1134,
    1138,
    1139,
    1142,
    1146,
    1161,
    1166,
    1169,
    1178,
    1180,
    1181,
    1182,
    1205,
    1211,
    1212,
    1216,
    1217,
    1218,
    1219,
    1220,
    1235,
    1236,
    1237,
    1238,
    1240,
    1241,
    1242,
    1369,
    1370,
    1371,
    1372,
    1373,
    1374,
    1375,
    1376,
    1377,
    1444,
    1453,
    1460,
    1461,
    1462,
    1471,
    1479,
    1485,
    1486,
    1487,
    1489,
    1494,
    1496,
    1502,
    1504,
    1507,
    1547,
    1558,
    1566,
    1590,
    1597,
    4134,
    4135,
    4137,
    4154,
    4534,
    23512,
    23517,
    40514,
    40515,
    40517,
    40518,
    40590,
    40592,
    40593,
    40594,
    40595,
    40596,
    40597,
    40645,
    40646,
    40647,
    40648,
    40649,
    40650,
    40666,
    40680,
    40701,
    40702,
    40704,
    40706,
    40713,
    40900,
    40910,
    40922,
    40978,
    40983,
    40999,
    41005,
    41007,
    41026,
    41142,
    41143,
    41144,
    41145,
    41146,
    41150,
    41156,
    41158,
    41159,
    41161,
    41228,
    41526,
    41672,
    41946,
    41964,
]


class colorize:
    """colorize strings in a bash shell"""

    def red(text):
        return f"\033[1;91m{text}\033[39m"

    def green(text):
        return f"\033[1;92m{text}\033[39m"

    def yellow(text):
        return f"\033[1;93m{text}\033[39m"

    def blue(text):
        return f"\033[1;94m{text}\033[39m"

    def magenta(text):
        return f"\033[1;95m{text}\033[39m"

    def cyan(text):
        return f"\033[1;96m{text}\033[39m"

    def white(text):
        return f"\033[1;97m{text}\033[39m"


def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])


# ------------------------- Logging configuration -------------------------
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(f"[{PIPELINE_TYPE}] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def log(s):
    """
    Prints a string surrounded by ###; Useful for debugging
    Parameters
    ----------
    s : str
        string for debugging
    """
    print("######################################################################")
    print(s)
    print("######################################################################")


# ------------------------- Main binary pipeline functions  -------------------------


def get_datasets_by_type(dataset_type):
    """
    Get datasets of a given type from the openml API endpoint, filtering them
    based on the DATASET_PARAMETERS dictionary constant.
    Parameters
    ----------
    # TODO: update this whenever I add more classifiers...
    dataset_type : str
        A string with the type of the machine learning problem to return.
        Currently the only one available is "BINARY", which will return
        classification problems with two classes only.
    """
    openml_df = openml.datasets.list_datasets(output_format="dataframe")[
        [
            "did",
            "name",
            "NumberOfInstances",
            "NumberOfFeatures",
            "NumberOfClasses",
            "NumberOfInstancesWithMissingValues",
        ]
    ].query(
        f"NumberOfInstances >= {DATASET_PARAMETERS.min_num_instances} & \
              NumberOfInstances <= {DATASET_PARAMETERS.max_num_instances} & \
              NumberOfFeatures >= {DATASET_PARAMETERS.min_num_features} & \
              NumberOfClasses <= {DATASET_PARAMETERS.max_num_classes} & \
             NumberOfInstancesWithMissingValues <= {DATASET_PARAMETERS.max_nan_percentage}*NumberOfInstances"
    )
    logger.info(f"OpenML filtered dataset has shape: {openml_df.shape}")
    if dataset_type == "BINARY":
        binary_datasets = openml_df.query("NumberOfClasses == 2")
        logger.info(f"Returning binary datasets, shape: {binary_datasets.shape}")
        return binary_datasets.sort_values(by="did")
    else:
        raise AttributeError("dataset_type is invalid!")


def build_full_dataset(openml_dataset, target_col_name):
    """
    Builds and return the full pandas dataframe representation of an openml_dataset,
    alongside important information like the attribute names, if there's a categorical
    feature in the dataset, etc.
    Parameters
    ----------
    openml_dataset : openml dataset
        The openml dataset representation
    target_col_name : str
        Name to rename the target column
    """
    logger.info(f"\tdataset name: {openml_dataset.name}")
    logger.info(f"\tdataset default target: {openml_dataset.default_target_attribute}")
    X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
        target=openml_dataset.default_target_attribute, dataset_format="dataframe"
    )
    assert X.shape[0] == y.shape[0]
    full_dataset = pd.concat(
        (X, pd.DataFrame(y).rename(columns={k: target_col_name for k in [y.name]})), axis=1
    )
    assert full_dataset.shape[0] == X.shape[0]
    label_categorizer_map = {}
    # if the >>target<< is not converted to int, use label_categorizer
    try:
        full_dataset = full_dataset.astype({target_col_name: "int"})
    except ValueError:
        _, full_dataset, label_logs = label_categorizer(
            columns_to_categorize=[target_col_name], store_mapping=True
        )(full_dataset)
        label_categorizer_map = fp.merge(
            label_logs["label_categorizer"]["mapping"], label_categorizer_map
        )

    return dotdict(
        dict(
            full_dataset=full_dataset,
            categorical_indicator=categorical_indicator,
            attribute_names=attribute_names,
            original_features=X.columns,
            label_categorizer_map=label_categorizer_map,
        )
    )


def analyze_dataset_and_process(
    dataset, original_features, target_col_name, attribute_names, categorical_indicator
):
    """
    Analyze the provided binary dataset, and remove some features according to
    the categorical indicator. The removed features are features that are categorical
    when classified by the analyzer function, but are not indicated in the
    categorical_indicator variable from openml. Will return a dataframe with the
    statistics and basic analysis done and some flags for categorical variables

    Parameters
    ----------
    dataset : pd.DataFrame
        Pandas dataframe of the full dataset
    original_features : np.array
        Original features of the model
    target_col_name : str
        Name of the target column
    attribute_names : List[str]
        List of strings with the name of the attributes of the dataframe (not exactly the features)
    """

    analyzer_fn = dataset_analyzer(
        feature_columns=original_features,
        target_columns=[target_col_name],
        id_columns=[],
        timestamp_columns=[],
    )

    logs = analyzer_fn(dataset)
    df_stats_original = statistics_dict_to_df(logs["statistics"])
    # filter out columns which are classified as categorical but are not in the categorical_indicator
    df_stats = df_stats_original.drop(
        columns=(
            set(
                df_stats_original.T.query("var_type == 'Categorical'")[
                    ~df_stats_original.T.query("var_type == 'Categorical'").index.isin(
                        np.array(attribute_names)[categorical_indicator]
                    )
                ].index.values
            )
            - set([target_col_name])
        )
    )
    # calculate feature_set and target
    feature_set = list(df_stats.drop(columns=target_col_name).columns.values)
    target = target_col_name
    return dotdict(
        dict(
            df_stats=df_stats,
            feature_set=feature_set,
            target=target,
            contains_categorical="Categorical" in df_stats[feature_set].T.var_type.values,
            categorical_features=df_stats.T.query("var_type == 'Categorical'").T.columns.values,
        )
    )


@fp.curry
def hyperparameter_producer(
    hyperparameter_name,
    values=None,
    generate_fn=None,
    length=None,
    base_producer=None,
    all_combinations=False,
):
    """
    Returns a function that builds a list with dictionaries with values for hyperparameters.
    You can either pass the values list directly using `values`, or pass a generator function and a
    length parameter.
    If a base_producer is provided, it will generate the combinations of hyperparameters from
    the base_producer and the new one being built, e.g if the base_producer has {"max_depth", "colsample"}
    and the new hyperparameter_name is "num_leaves", it will generate a list with the values like:
        [{"num_leaves"},
         {"max_depth", "colsample", "num_leaves"}]
    if the all_combinations flag is true, it will generate all possible combinations, e.g.:
        [{"num_leaves"},
         {"max_depth", "num_leaves"},
         {"colsample", "num_leaves"},
         {"max_depth", "colsample", "num_leaves"}]

    Parameters
    ----------
    hyperparameter_name : str
        the name of the hyperparameter to generate
    values: np.array or list
        a list of values for the hyperparameter
    generate_fn : lambda function
        lambda function that when called generate one possible value of a hyperparameter
    length : int
        size of the list of values for the hyperparameter
    base_producer: hyperparameter_producer
        if it's passed, it will be used as a base to generate the combinations of hyperparameters
    all_combinations: boolean
        flag to indicate if all possible combinations fo hyperparameters will be generated
    """

    # must pass both or pass none
    assert not ((generate_fn is None) ^ (length is None))
    if generate_fn is not None:
        h_values = [generate_fn() for _ in range(length)]
    elif values is not None:
        h_values = values
    else:
        raise ValueError("You need to provide at least one hyperparameter base value!!")

    def gen_space():
        base_dict = {} if base_producer is None else fp.last(base_producer())
        h_space = {hyperparameter_name: h_values}
        space_list = []
        full_space_list = []
        if all_combinations and len(base_dict.keys()) >= 2:
            space_list = [fp.merge({key: val}, h_space) for key, val in base_dict.items()]
        if base_producer is not None:
            full_space_list = [fp.merge(base_dict, h_space)]

        return [h_space] + space_list + full_space_list

    return gen_space


def binary_hyperparameter_space(
    dataset, num_estimators_length=20, lr_length=20, max_depth_length=20, seed=None
):
    """
    Construct a hyperparameter space for binary classification problems. This hyperparameter tree contains
    the following structure:
                        learning_rate
                              |
                              |
                        num_estimators
                              |
                              |
                          max_depth

    The specific space for each of them can be checked in the plots description, but basically
    some of them have a fixed distribution e.g. max_depth, and others like num_estimators depend
    on the actual size of the dataset (dsize).

    Parameters
    ----------
    dataset : DataFrame
        the original full dataset
    num_estimators_length : int
        Number of num_estimators values to generate
    lr_length : int
        Number of learning_rate values to generate
    max_depth_length : int
        Number of max_depth values to generate
    seed : int
        Seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    def max_value_num_estimators(dsize):
        # function that defines the maximum num_estimators value depending on sample size
        left_limit = 10000
        right_limit = 300000
        left_constant = 400
        right_constant = 1700
        a = (right_constant - left_constant) / (right_limit - left_limit)
        b = left_constant - a
        if dsize <= left_limit:
            return left_constant
        elif dsize >= right_limit:
            return right_constant
        else:
            return a * dsize + b

    # generator function for num_estimator hyperparameter
    def gen_num_estimators(dsize, max_value_fn, num=20):
        return np.linspace(start=min(dsize / 4, 50), stop=max_value_fn(dsize), num=num, dtype=int)

    # generator function for learning rate hyperparameter
    def gen_lr(dsize, num=20):
        middle_value = max(0.01, 0.1 * min(1, dsize / 10000))
        half_num = max(num // 2, 2)  # at least two
        return np.unique(
            np.concatenate(
                (
                    [
                        mid / (i + 1)
                        for i, mid in enumerate(np.repeat(middle_value, repeats=half_num))
                    ],
                    [
                        (i + 1) * mid
                        for i, mid in enumerate(np.repeat(middle_value, repeats=half_num))
                    ],
                )
            )
        )

    lr_prod = hyperparameter_producer(
        hyperparameter_name="learning_rate", values=gen_lr(dsize=dataset.shape[0], num=lr_length)
    )

    num_est_prod = hyperparameter_producer(
        hyperparameter_name="num_estimators",
        values=gen_num_estimators(
            dsize=dataset.shape[0], max_value_fn=max_value_num_estimators, num=num_estimators_length
        ),
        base_producer=lr_prod,
        all_combinations=True,
    )

    max_depth_prod = hyperparameter_producer(
        hyperparameter_name="max_depth",
        values=np.arange(3, 3 + max_depth_length),
        base_producer=num_est_prod,
        all_combinations=True,
    )
    hyperparameter_tree = (lr_prod, num_est_prod, max_depth_prod)

    return hyperparameter_tree


def dataset_split(dataset, feature_set, target, train_percentage, seed=42):
    """
    Split a dataset and returns the train and test set
    Parameters
    ----------
    dataset : pd.DataFrame
        Pandas dataframe of the full dataset
    feature_set : list
        Features of the model
    target_col_name : str
        Name of the target column
    train_percentage : float
        Percentage of the dataset that will be used for training
    """
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[feature_set], dataset[[target]], train_size=train_percentage, random_state=seed
    )
    train_set = pd.concat((x_train, y_train), axis=1)
    test_set = pd.concat((x_test, y_test), axis=1)

    del x_train, x_test, y_train, y_test
    gc.collect()

    return train_set, test_set


def get_eval_fn_binary_classifier(evaluation_target, prediction_column):
    """
    Build evaluation function for a binary classifier problem,
    which consist of an AUC evaluator, a Logloss evaluator and a Brier score
    evaluator

    Parameters
    ----------
    evaluation_target : str
        The actual target (ground truth) of the dataset
    prediction_column : str
        The prediction column of the model
    """

    return combined_evaluators(
        evaluators=[
            auc_evaluator(target_column=evaluation_target, prediction_column=prediction_column),
            logloss_evaluator(target_column=evaluation_target, prediction_column=prediction_column),
            brier_score_evaluator(
                target_column=evaluation_target, prediction_column=prediction_column
            ),
        ]
    )


def get_eval_fn(evaluation_target, prediction_column, model_type):
    """
    Build evaluation function for a given model_type
    Parameters
    ----------
    # TODO: update this whenever we add more classifiers
    model_type : str
        A string with the type of the machine learning problem to return.
        Currently the only one available is "BINARY", which will return
        classification problems with two classes only.
    evaluation_target : str
        The actual target (ground truth) of the dataset
    prediction_column : str
        The prediction column of the model
    """
    if model_type == "BINARY":
        return get_eval_fn_binary_classifier(
            evaluation_target=evaluation_target, prediction_column=prediction_column
        )
    else:
        raise AttributeError("model type is invalid!")


@fp.curry
def run_training_binary(
    dataset,
    features,
    target,
    prediction_column,
    extra_params=None,
    num_estimators=None,
    learning_rate=None,
    contains_categorical=None,
    columns_to_categorize=None,
):
    """
    Run a single training of a lgbm_classifier model. This returns the basic applied learner,
    which is the predict function, the scored dataset and the logs from the pipeline
    Parameters
    ----------
    dataset : pd.DataFrame
        Pandas dataframe of the full dataset
    features : list
        Features of the model
    target : str
        target column
    prediction_column : str
        name of the prediction column
    extra_params : dict
        dict with the extra params of the lgbm_classifier
    num_estimators : float
        number of boosting rounds
    learning_rate : float
        learning rate to test
    contains_categorical : bool
        boolean to indicate if the current dataset contains categorical values
    columns_to_categorize : list like
        List of the names of categorical features
    """
    default_params = HYPERPARAMETERS_CONFIG.default_lgbm_main_params
    num_estimators = default_params["num_estimators"] if num_estimators is None else num_estimators
    learning_rate = default_params["learning_rate"] if learning_rate is None else learning_rate

    # if there's at least one categorical feature use the label_categorizer
    if contains_categorical:
        lgbm_pipeline = build_pipeline(
            label_categorizer(
                columns_to_categorize=[str(c) for c in columns_to_categorize], store_mapping=True
            ),
            lgbm_classification_learner(
                features=features,
                target=target,
                prediction_column=prediction_column,
                num_estimators=num_estimators,
                learning_rate=learning_rate,
                extra_params=extra_params,
            ),
        )
    else:
        lgbm_pipeline = build_pipeline(
            lgbm_classification_learner(
                features=features,
                target=target,
                prediction_column=prediction_column,
                num_estimators=num_estimators,
                learning_rate=learning_rate,
                extra_params=extra_params,
            )
        )

    return lgbm_pipeline(dataset)


def retrieve_extra_params(remaining_params):
    """
    Auxiliar function that returns the extra params if available, alongside the
    extra_params (default_lgbm_extra_params)
    Parameters
    ----------
    remaining_params : dict
        dict with the possible extra params
    """
    if (
        remaining_params is not None
        and "max_depth" in remaining_params
        and "num_leaves" not in remaining_params
    ):
        remaining_params = fp.merge(
            remaining_params, dict(num_leaves=min(2 ** 15, 2 ** remaining_params["max_depth"] - 1))
        )
    return fp.merge(remaining_params, HYPERPARAMETERS_CONFIG.default_lgbm_extra_params)


@fp.curry
def binary_model_experiment(
    dataset, hyperparameter_tree, analyzer_info, prediction_column, eval_fn
):
    """
    Returns a function that will run the full binary experiment on the dataset
    and return the logs after the evaluation
    Parameters
    ----------
    dataset : dict
        dict with the train and test datasets
    hyperparameter_tree : list of dicts
        The tree that contains the hyperparameter space
    analyzer_info : dict
        Dictionary with the information of the dataset
    prediction_column : string
        Name of the column with the model prediction output
    eval_fn : curried function
        Evaluator function to apply on the dataset
    """

    @fp.curry
    def train_eval_fn(dataset, eval_fn, hp_dict):
        """
        Receives a dictionary of hyperparameters, run the experiments and return the logs of each run!

        Parameters
        ----------
        dataset : dict
            dict with the train and test datasets
        eval_fn : curried function
            Evaluator function to apply on the dataset
        hp_dict : dict
            Single dictionary with hyperparameters to run the experiments (single level
            dictionary)
        """
        hp_names = hp_dict.keys()
        lr_name = "learning_rate"
        num_est_name = "num_estimators"
        experiment_logs = []
        get_last_param = fp.compose(fp.last, fp.last)

        def get_lr(_):
            return None

        def get_num_est(_):
            return None

        def filter_extra_params(hp):
            return dict(filter(lambda t: t[0] != lr_name and t[0] != num_est_name, hp))

        if lr_name in hp_names:

            def get_lr(hp):
                return get_last_param(filter(lambda t: t[0] == lr_name, hp))

        if num_est_name in hp_names:

            def get_num_est(hp):
                return get_last_param(filter(lambda t: t[0] == num_est_name, hp))

        logger.info(colorize.cyan(f"\n\t\t Running current hyperparameter space: {hp_dict}"))
        for hp_combination in product(
            *(zip(np.repeat([k], len(v)), v) for k, v in hp_dict.items())
        ):
            lr, num_est = get_lr(hp_combination), get_num_est(hp_combination)
            extra_params = retrieve_extra_params(filter_extra_params(hp_combination))
            model, scored_df, logs = run_training_binary(
                dataset=dataset.train_set,
                features=analyzer_info.feature_set,
                target=analyzer_info.target,
                prediction_column=prediction_column,
                num_estimators=num_est,
                learning_rate=lr,
                extra_params=extra_params,
                contains_categorical=analyzer_info.contains_categorical,
                columns_to_categorize=analyzer_info.categorical_features,
            )

            train_result = dict(train_result=eval_fn(scored_df))
            test_result = dict(test_result=eval_fn(model(dataset.test_set)))
            training_time = logs["lgbm_classification_learner"]["running_time"]
            logger.info(colorize.yellow(f"\n\t\t\t {training_time}"))
            del scored_df
            gc.collect()

            experiment_logs.append(
                fp.merge(
                    fp.merge(dict(hp_combination), extra_params),
                    train_result,
                    test_result,
                    dict(training_time=training_time),
                )
            )
        return experiment_logs

    logs = fp.pipe(
        hyperparameter_tree,
        fp.curried.map(lambda hp: hp()),
        flat_list,
        fp.curried.map(
            lambda hp: train_eval_fn(eval_fn=eval_fn, hp_dict=hp)
        ),  # function expects a dataset
        fp.curried.map(lambda run: run(dataset=dataset)),
        flat_list,
    )
    return logs


def save_experiment(model_type, did, openml_object, analyzer_info, shapes, hp_tree, final_result):
    """
    Save the final experiment in a folder with a given dataset id  (did), in different
    pickle files for each object.
    Parameters
    ----------
    model_type : str
        String representing the model type
    did : int
        the dataset id
    openml_object : openml dataset
        Openml object
    analyzer_info : dict
        The analyzer info dict containing the basic automatic statistical analysis of a dataframe
    shapes : tuple
        Tuple representing the shape of the dataframe
    hp_tree : tuple of functions
        Tuple where each element is a leaf of a hyperparameter space
    final_result : list of dicts
        List of dicts containg the evaluation of each run of a model training
    """
    base_bath = Path(model_type)
    base_bath.mkdir(exist_ok=True)
    model_path = Path(model_type + "/" + str(did))
    model_path.mkdir(exist_ok=True)
    namepath = model_type + "/" + str(did) + "/"

    open(namepath + "openml_object.pkl", "wb").write(cloudpickle.dumps(openml_object))
    open(namepath + "analyzer_info.pkl", "wb").write(
        cloudpickle.dumps(
            dict(
                feature_set=analyzer_info.feature_set,
                target=analyzer_info.target,
                contains_categorical=analyzer_info.contains_categorical,
                categorical_features=analyzer_info.categorical_features,
            )
        )
    )
    open(namepath + "shape.pkl", "wb").write(cloudpickle.dumps(shapes))
    open(namepath + "hp_tree.pkl", "wb").write(cloudpickle.dumps(hp_tree))
    open(namepath + "final_result.pkl", "wb").write(cloudpickle.dumps(final_result))
    analyzer_info["df_stats"].to_pickle(namepath + "df_stats.pkl")


def get_calculated_dids(model_type):
    """
    Returns a set with all the calculated dataset ids. This is useful to do a warm start,
    the program doesn't need to run again the same did two times!
    If there's no base folder, it returns an empty set
    Parameters
    ----------
    model_type : str
        string representing the model type
    """
    base_path = Path(model_type)
    if base_path.is_dir():
        return set(
            (int(child.__str__().replace(model_type + "/", "")) for child in base_path.iterdir())
        )
    return set()


def binary_pipeline(dataset, target_col_name="target", output_prediction_column="prediction"):
    """
    Main function that defines a binary classification task experiment. The pipeline is defined as:
        1 -> build dataset
        2 -> analyze dataset
        3 -> create hyperparameter space tree
        4 -> split dataset
        5 -> create evaluator function
        6 -> run final model experiment
        7 -> save all the outputs in a specific folder
    Parameters
    ----------
    dataset : openml dataset object
        OpenML object which represents a dataset
    target_col_name : string
        the name of the target column
    output_prediction_column : str
        name of the prediction column of the classifier
    """
    logger.info(colorize.green(f"[1] Building dataset..."))
    dataset_info = build_full_dataset(dataset, target_col_name=target_col_name)
    full_dataset = dataset_info.full_dataset
    logger.info(f"\tdataset shape: {full_dataset.shape}")
    logger.info(f"\tdataset columns: {full_dataset.columns}")

    logger.info(colorize.green(f"[2] Analyzing dataset..."))
    analyzer_info = analyze_dataset_and_process(
        full_dataset,
        original_features=dataset_info.original_features,
        target_col_name=target_col_name,
        attribute_names=dataset_info.attribute_names,
        categorical_indicator=dataset_info.categorical_indicator,
    )

    logger.info(colorize.green(f"[3] Building hyperparamer distribution..."))
    # TODO stuf here
    hp_tree = binary_hyperparameter_space(
        full_dataset,
        # num_estimators_length=HYPERPARAMETERS_CONFIG.num_estimators_length,
        # lr_length=HYPERPARAMETERS_CONFIG.lr_length,
        # max_depth_length=HYPERPARAMETERS_CONFIG.max_depth_length,
          num_estimators_length=2,
          lr_length=2,
          max_depth_length=1,
        seed=RND_STATE_SEED,
    )

    logger.info(colorize.green(f"[4] Splitting dataset..."))
    train_set, test_set = dataset_split(
        full_dataset,
        feature_set=analyzer_info.feature_set,
        target=analyzer_info.target,
        train_percentage=DATASET_PARAMETERS.train_percentage,
        seed=RND_STATE_SEED,
    )
    logger.info(f"\t train_set and test_set shape: {train_set.shape, test_set.shape}")
    # TODO stuff here
    return {}
    logger.info(colorize.green(f"[5] Building evaluator function..."))
    eval_fn = get_eval_fn(
        evaluation_target=analyzer_info.target,
        prediction_column=output_prediction_column,
        model_type="BINARY",
    )

    logger.info(colorize.green(f"[6] Running model experiment..."))
    final_result = binary_model_experiment(
        dataset=dotdict(dict(train_set=train_set, test_set=test_set)),
        hyperparameter_tree=hp_tree,
        analyzer_info=analyzer_info,
        prediction_column=output_prediction_column,
        eval_fn=eval_fn,
    )

    logger.info(colorize.green(f"[7] Saving logs..."))
    save_experiment(
        model_type="BINARY",
        did=dataset.dataset_id,
        openml_object=dataset,
        analyzer_info=analyzer_info,
        shapes=dict(train_set=train_set.shape, test_set=test_set.shape),
        hp_tree=hp_tree,
        final_result=final_result,
    )
    logger.info(colorize.magenta("Finished Training!"))


def custom_did_by_machine(did_dfs, max_datasets=30):
    # Distribute the processing between different machines in vision network
    hostname = socket.gethostname()
    if hostname == "tolstoi":
        return did_dfs[-max_datasets:]
    elif hostname == "hulk1":
        return did_dfs[-2 * max_datasets: -max_datasets]
    elif hostname == "hulk3":
        return did_dfs[-3 * max_datasets: -2 * max_datasets]
    elif hostname == "hulk4":
        return did_dfs[-4 * max_datasets: -3 * max_datasets]
    elif hostname == "laplace":
        return did_dfs[-5 * max_datasets: -4 * max_datasets]
    else:
        return did_dfs


def notify_me(did, datasets, custom_title="Started Run"):
    hostname = socket.gethostname()
    # Send text message
    bot_token = "<put your token here>"
    bot_chatID = "<put the chat ID here>"
    bot_message = f"""------ [{hostname}] {custom_title} ------\n
    DID = {did}
    NumberOfInstances = {datasets.query(f'did == {did}')['NumberOfInstances'].values[0]}
    """
    send_text = (
        "https://api.telegram.org/bot"
        + bot_token
        + "/sendMessage?chat_id="
        + bot_chatID
        + "&parse_mode=Markdown&text="
        + bot_message
    )
    requests.get(send_text)


def run_binary_pipeline():
    """
    Fetch datasets by binary type, and run a binary_pipeline for each of them.
    This is the 'main' function to be run in this file.
    """
    # How many datasets to load and run experiments
    MAX_DATASETS_TO_FETCH = 30
    datasets = get_datasets_by_type(dataset_type="BINARY")
    print_valid_datasets(datasets)
    exit()
    # I use a warming start to avoid redundant computation on the same dataset
    already_calculated = get_calculated_dids(model_type="BINARY")
    # did_vals = custom_did_by_machine(VALID_DATASETS_DID, max_datasets=MAX_DATASETS_TO_FETCH)

    for did in did_vals:
        if did in already_calculated:
            logger.info(colorize.yellow(f" Skipping dataset... DID - {did}"))
            continue
        logger.info(f"running binary pipeline for dataset with DID: {did}")
        try:
            current_dataset = openml.datasets.get_dataset(int(did))
            # notify_me(did, datasets) # TODO
            binary_pipeline(dataset=current_dataset)
        except arff.BadAttributeType:
            logger.info(colorize.red("Error while reading dataset... Continue"))


def print_valid_datasets(datasets):
    these_are_valid = []
    for did in datasets.did.values:
        try:
            _ = openml.datasets.get_dataset(int(did))
            these_are_valid.append(did)
        except Exception as e:
            print(f"{did} - FAILED")
            print(e)
    print(these_are_valid)


def offline_test():
    """
    Simple debug function that creates a sample dataset with toy data. Useful when
    developing new functionalities
    This doesn't run the full binary pipeline
    """
    np.random.seed(RND_STATE_SEED)
    full_dataset = pd.DataFrame(
        dict(a=np.random.random(1000), target=np.random.choice([0, 1], size=1000))
    )
    hp_space = binary_hyperparameter_space(
        full_dataset, num_estimators_length=2, lr_length=2, max_depth_length=1, seed=RND_STATE_SEED
    )

    info = dotdict(
        dict(
            df_stats=pd.DataFrame({"oi": [11]}),
            feature_set=["a"],
            target="target",
            contains_categorical=False,
            categorical_features=None,
        )
    )
    train_set, test_set = dataset_split(
        full_dataset,
        feature_set=info.feature_set,
        target=info.target,
        train_percentage=DATASET_PARAMETERS.train_percentage,
        seed=RND_STATE_SEED,
    )

    res = binary_model_experiment(
        dataset=dotdict(dict(train_set=train_set, test_set=test_set)),
        hyperparameter_tree=hp_space,
        analyzer_info=info,
        prediction_column="prediction",
        eval_fn=auc_evaluator(target_column="target", prediction_column="prediction"),
    )


if __name__ == "__main__":
    run_binary_pipeline()
