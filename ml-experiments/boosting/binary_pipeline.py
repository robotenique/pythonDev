import gc
import logging
import sys
from multiprocessing import cpu_count

import numpy as np
import openml
import pandas as pd
import toolz as fp
from fklearn.training.classification import lgbm_classification_learner
from fklearn.training.exploration import (dataset_analyzer,
                                          statistics_dict_to_df)
from fklearn.validation.evaluators import (
    combined_evaluators,
    auc_evaluator,
    logloss_evaluator,
    brier_score_evaluator,
)
from fklearn.training.pipeline import build_pipeline
from fklearn.training.transformation import label_categorizer
from sklearn.model_selection import train_test_split

# ----------------------------- Global Contants ---------------------------


RND_STATE_SEED = 42
PIPELINE_TYPE = "Binary Classifier"

DATASET_PARAMETERS = dict(
    min_num_features=3,
    min_num_instances=1000,
    max_num_classes=100,
    max_nan_percentage=.2,
    train_percentage=.7
)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class colorize():
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

    def White(text):
        return f"\033[1;97m{text}\033[39m"


# ------------------------- Logging configuration -------------------------
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(f"[{PIPELINE_TYPE}] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
            auc_evaluator(
                target_column=evaluation_target, prediction_column=prediction_column
            ),
            logloss_evaluator(
                target_column=evaluation_target, prediction_column=prediction_column
            ),
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


def get_datasets_by_type(dataset_type):
    """
    Get datasets of a given type from the openml API endpoint, filtering them
    based on the DATASET_PARAMETERS dictionary constant.
    Parameters
    ----------
    # TODO: update this whenever we add more classifiers
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
    ].query(f"NumberOfInstances >= {DATASET_PARAMETERS['min_num_instances']} & \
              NumberOfFeatures >= {DATASET_PARAMETERS['min_num_features']} & \
              NumberOfClasses <= {DATASET_PARAMETERS['max_num_classes']} & \
             NumberOfInstancesWithMissingValues <= {DATASET_PARAMETERS['max_nan_percentage']}*NumberOfInstances")

    logger.info(f"OpenML filtered dataset has shape: {openml_df.shape}")
    if dataset_type == "BINARY":
        binary_datasets = openml_df.query("NumberOfClasses == 2")
        logger.info(f"Returning binary datasets, shape: {binary_datasets.shape}")
        return binary_datasets
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
        (
            X,
            pd.DataFrame(y)
            .rename(columns={k: target_col_name for k in [y.name]}),
        ),
        axis=1,
    )
    assert full_dataset.shape[0] == X.shape[0]
    label_categorizer_map = {}
    # if the target is not converted to int, use label_categorizer
    try:
        full_dataset = full_dataset.astype({target_col_name: "int"})
    except ValueError:
        _, full_dataset, label_logs = label_categorizer(columns_to_categorize=[target_col_name],
                                                        store_mapping=True)(full_dataset)
        label_categorizer_map = fp.merge(label_logs["label_categorizer"]["mapping"], label_categorizer_map)

    return dotdict(
        dict(full_dataset=full_dataset,
             categorical_indicator=categorical_indicator,
             attribute_names=attribute_names,
             original_features=X.columns,
             label_categorizer_map=label_categorizer_map
             )
    )


def analyze_dataset_and_process(dataset, original_features, target_col_name, attribute_names, categorical_indicator):
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
            set(df_stats_original.T.query("var_type == 'Categorical'")[
                ~df_stats_original.T.query("var_type == 'Categorical'").index.isin(
                    np.array(attribute_names)[categorical_indicator]
                )
            ].index.values) - set([target_col_name])
        )
    )
    # calculate feature_set and target
    feature_set = list(df_stats.drop(columns=target_col_name).columns.values)
    target = target_col_name
    return dotdict(
        dict(df_stats=df_stats,
             feature_set=feature_set,
             target=target,
             contains_categorical='Categorical' in df_stats[feature_set].T.var_type.values,
             categorical_features=df_stats.T.query("var_type == 'Categorical'").T.columns.values
             )
    )


def run_training_binary(dataset, features, target, prediction_column, num_estimators, learning_rate, extra_params,
                        contains_categorical=None, columns_to_categorize=None):
    # must pass both or pass none
    assert not((contains_categorical is None) ^ (columns_to_categorize is None))
    # if there's at least one categorical feature use the label_categorizer
    if contains_categorical:
        lgbm_pipeline = build_pipeline(
            label_categorizer(
                columns_to_categorize=columns_to_categorize,
                store_mapping=True,
            ),
            lgbm_classification_learner(
                features=features, target=target, prediction_column=prediction_column,
                num_estimators=num_estimators, learning_rate=learning_rate, extra_params=extra_params),
        )
    else:
        lgbm_pipeline = build_pipeline(
            lgbm_classification_learner(
                features=features, target=target, prediction_column=prediction_column,
                num_estimators=num_estimators, learning_rate=learning_rate, extra_params=extra_params
            ),
        )
    return lgbm_pipeline(dataset)


def binary_pipeline(dataset, target_col_name="target", output_prediction_column="prediction"):
    logger.info(colorize.green(f"[1] Building dataset..."))
    dataset_info = build_full_dataset(dataset, target_col_name=target_col_name)
    full_dataset = dataset_info.full_dataset
    label_categorizer_map = dataset_info.label_categorizer_map
    logger.info(f"\tdataset shape: {full_dataset.shape}")
    logger.info(f"\tdataset columns: {full_dataset.columns}")

    logger.info(colorize.green(f"[2] Analyzing dataset..."))
    analyzer_info = analyze_dataset_and_process(full_dataset, original_features=dataset_info.original_features,
                                                target_col_name=target_col_name,
                                                attribute_names=dataset_info.attribute_names,
                                                categorical_indicator=dataset_info.categorical_indicator)

    logger.info(colorize.green(f"[2] Building hyperparamer distribution..."))
    # lgbm hyperparameters
    learning_rate = 0.1
    num_estimators = 30
    other_lgbm_params = {
        "max_depth": 3,
        "seed": RND_STATE_SEED,
        "min_gain_to_split": 0.5,
        "nthread": cpu_count(),
        "verbose": -1
    }
    
    logger.info("\tsplitting dataset...")
    x_train, x_test, y_train, y_test = train_test_split(
        full_dataset[feature_set],
        full_dataset[[target]],
        train_size=DATASET_PARAMETERS["train_percentage"],
        random_state=RND_STATE_SEED,
    )
    train_set = pd.concat((x_train, y_train), axis=1)
    test_set = pd.concat((x_test, y_test), axis=1)

    del x_train, x_test, y_train, y_test
    gc.collect()
    logger.info(f"\t train_set and test_set shape: {train_set.shape, test_set.shape}")
    logger.info("\tCreating evaluation function...")
    eval_fn = get_eval_fn(evaluation_target=target_col_name, prediction_column=output_prediction_column,
                          model_type="BINARY")

    logger.info("\ttraining models...")
    predict_fn, scored_dataset, training_logs = lgbm_pipeline(train_set)
    train_result = eval_fn(scored_dataset)
    test_result = eval_fn(predict_fn(test_set))
    print(train_result)
    print(test_result)
    logger.info("\tfinished training")


# TODO: generalize this pipeline to other types of classification problems
def run_binary_pipeline():
    datasets = get_datasets_by_type(dataset_type="BINARY")
    MAX_DATASETS_TO_FETCH = 1
    did_vals = datasets.did.values[:MAX_DATASETS_TO_FETCH]
    for did in did_vals:
        logger.info(f"running binary pipeline for dataset with DID: {did}")
        binary_pipeline(dataset=openml.datasets.get_dataset(int(did)))
    # print(datasets.head())


if __name__ == "__main__":
    run_binary_pipeline()
