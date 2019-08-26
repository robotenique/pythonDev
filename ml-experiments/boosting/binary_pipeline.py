import functools
import gc
import logging
import operator
from itertools import product, starmap
from multiprocessing import cpu_count

import numpy as np
import openml
import pandas as pd
import toolz as fp
from fklearn.training.classification import lgbm_classification_learner
from fklearn.training.exploration import (dataset_analyzer,
                                          statistics_dict_to_df)
from fklearn.training.pipeline import build_pipeline
from fklearn.training.transformation import label_categorizer
from fklearn.validation.evaluators import (auc_evaluator,
                                           brier_score_evaluator,
                                           combined_evaluators,
                                           logloss_evaluator)
from sklearn.model_selection import train_test_split


# TODO: do a warm start
def log(s):
    print("######################################################################")
    print(s)
    print("######################################################################")
# ----------------------------- Global Contants ---------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


RND_STATE_SEED = 42
PIPELINE_TYPE = "Binary Classifier"

DATASET_PARAMETERS = dotdict(dict(
    min_num_features=3,
    min_num_instances=1000,
    max_num_classes=100,
    max_num_instances=5000000,
    max_nan_percentage=.2,
    train_percentage=.7
))

HYPERPARAMETERS_CONFIG = dotdict(dict(
    num_estimators_length=20,  # number of num_estimator values to generate
    lr_length=6,  # number of learning rate values to generate
    max_depth_length=20,  # number of max_pdeth values to generate,
    default_lgbm_main_params = {
        "num_estimators": 30,
        "learning_rate": 0.1
    },
    default_lgbm_extra_params={
        "seed": RND_STATE_SEED,
        "nthread": cpu_count(),
        "verbose": -1
    }
))


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


def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])

# ------------------------- Logging configuration -------------------------
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(f"[{PIPELINE_TYPE}] %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
    ].query(f"NumberOfInstances >= {DATASET_PARAMETERS.min_num_instances} & \
              NumberOfInstances <= {DATASET_PARAMETERS.max_num_instances} & \
              NumberOfFeatures >= {DATASET_PARAMETERS.min_num_features} & \
              NumberOfClasses <= {DATASET_PARAMETERS.max_num_classes} & \
             NumberOfInstancesWithMissingValues <= {DATASET_PARAMETERS.max_nan_percentage}*NumberOfInstances")
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


@fp.curry
def hyperparameter_producer(hyperparameter_name, values=None, generate_fn=None, length=None,
                            base_producer=None, all_combinations=False):
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
    assert not((generate_fn is None) ^ (length is None))
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


def binary_hyperparameter_space(dataset, num_estimators_length=20, lr_length=20, max_depth_length=20, seed=None):
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
        return np.unique(np.concatenate(
            ([mid / (i + 1) for i, mid in enumerate(np.repeat(middle_value, repeats=half_num))],
             [(i + 1) * mid for i, mid in enumerate(np.repeat(middle_value, repeats=half_num))])
        ))

    lr_prod = hyperparameter_producer(hyperparameter_name="learning_rate",
                                      values=gen_lr(dsize=dataset.shape[0], num=lr_length))

    num_est_prod = hyperparameter_producer(hyperparameter_name="num_estimators",
                                           values=gen_num_estimators(dsize=dataset.shape[0],
                                                                     max_value_fn=max_value_num_estimators,
                                                                     num=num_estimators_length),
                                           base_producer=lr_prod,
                                           all_combinations=True)
    # TODO: max depth

    max_depth_prod = hyperparameter_producer(hyperparameter_name="max_depth",
                                             values=np.arange(3, 3 + max_depth_length),
                                             base_producer=num_est_prod,
                                             all_combinations=True)
    hyperparameter_tree = (lr_prod,
                           num_est_prod,
                           max_depth_prod)

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
        dataset[feature_set],
        dataset[[target]],
        train_size=train_percentage,
        random_state=seed,
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


@fp.curry
def run_training_binary(dataset, features, target, prediction_column, extra_params=None,
                        num_estimators=None, learning_rate=None, contains_categorical=None,
                        columns_to_categorize=None):
    default_params = HYPERPARAMETERS_CONFIG.default_lgbm_main_params
    # must pass both or pass none
    assert not((not contains_categorical) ^ (columns_to_categorize is None))
    num_estimators = default_params["num_estimators"] if num_estimators is None else num_estimators
    learning_rate = default_params["learning_rate"] if learning_rate is None else learning_rate
    # if there's at least one categorical feature use the label_categorizer
    # TODO: can I get the default value of keyargs?? then I just need to get them from the lgbm_classification_learner
    if contains_categorical:
        lgbm_pipeline = build_pipeline(
            label_categorizer(
                columns_to_categorize=[str(c) for c in columns_to_categorize],
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

    log(lgbm_pipeline)
    return lgbm_pipeline(dataset)


@fp.curry
def binary_model_experiment(dataset, hyperparameter_tree, analyzer_info, prediction_column, eval_fn):
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
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11")

        n_keys = len(hp_dict.keys())
        hp_names = hp_dict.keys()
        lr_name = "learning_rate"
        num_est_name = "num_estimators"
        get_last_param = fp.compose(fp.last, fp.last)
        def get_lr(_): return None
        def get_num_est(_): return None
        def filter_extra_params(hp):
            return filter(lambda t: t[0] != lr_name and t[0] != num_est_name, hp)
        if lr_name in hp_names:
            def get_lr(hp): return get_last_param(filter(lambda t: t[0] == lr_name, hp))
        if num_est_name in hp_names:
            def get_num_est(hp): return get_last_param(filter(lambda t: t[0] == num_est_name, hp))

        for hp_combination in product(*(zip(np.repeat([k], len(v)), v) for k, v in hp_dict.items())):
            
            lr, num_est = get_lr(hp_combination), get_num_est(hp_combination)

            remaining_params = dict(filter_extra_params(hp_combination))       
            extra_params = fp.merge(remaining_params,
                                    HYPERPARAMETERS_CONFIG.default_lgbm_extra_params)

            model, scored_df, logs = run_training_binary(
                dataset=dataset.train_set,
                features=analyzer_info.feature_set,
                target=analyzer_info.target,
                prediction_column=prediction_column,
                num_estimators=num_est,
                learning_rate=lr,
                extra_params=extra_params,
                contains_categorical=analyzer_info.contains_categorical,
                columns_to_categorize=analyzer_info.categorical_features
            )
            
            train_result = eval_fn(scored_df)
            test_result = eval_fn(model(dataset.test_set))
            print(train_result)
            print(test_result)
            exit()

    # TODO: Pipeline for training
    res = fp.pipe(hyperparameter_tree,
                  fp.curried.map(lambda hp: hp()),
                  flat_list,
                  fp.curried.map(lambda hp: train_eval_fn(eval_fn=eval_fn, hp_dict=hp)),  # function expect  a dataset
                  fp.curried.map(lambda run: run(dataset=dataset))
                  
                  )
    from pprint import pprint
    pprint(list(res))




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

    logger.info(colorize.green(f"[3] Building hyperparamer distribution..."))
    hp_tree = binary_hyperparameter_space(full_dataset,
                                          num_estimators_length=HYPERPARAMETERS_CONFIG.num_estimators_length,
                                          lr_length=HYPERPARAMETERS_CONFIG.lr_length,
                                          max_depth_length=HYPERPARAMETERS_CONFIG.max_depth_length,
                                          seed=RND_STATE_SEED)

    logger.info(colorize.green(f"[4] Splitting dataset..."))
    train_set, test_set = dataset_split(full_dataset,
                                        feature_set=analyzer_info.feature_set,
                                        target=analyzer_info.target,
                                        train_percentage=DATASET_PARAMETERS.train_percentage,
                                        seed=RND_STATE_SEED)
    logger.info(f"\t train_set and test_set shape: {train_set.shape, test_set.shape}")

    logger.info(colorize.green(f"[5] Building evaluator function..."))
    eval_fn = get_eval_fn(evaluation_target=analyzer_info.target,
                          prediction_column=output_prediction_column,
                          model_type="BINARY")

    logger.info(colorize.green(f"[6] Running  model experiment..."))
    binary_model_experiment(dataset=dotdict(dict(train_set=train_set, test_set=test_set)),
                            hyperparameter_tree=hp_tree,
                            analyzer_info=analyzer_info,
                            prediction_column=output_prediction_column,
                            eval_fn=eval_fn)
    logger.info(colorize.cyan("Finished Training!"))


# TODO: generalize this pipeline to other types of classification problems
def run_binary_pipeline():
    datasets = get_datasets_by_type(dataset_type="BINARY")
    MAX_DATASETS_TO_FETCH = 1
    did_vals = datasets.did.values[:MAX_DATASETS_TO_FETCH]
    for did in did_vals:
        logger.info(f"running binary pipeline for dataset with DID: {did}")
        binary_pipeline(dataset=openml.datasets.get_dataset(int(did)))



def offline_test():
    np.random.seed(RND_STATE_SEED)
    full_dataset = pd.DataFrame(
        dict(
            a=np.random.random(1000),
            target=np.random.choice([0, 1], size=1000)
        )
    )
    hp_space = binary_hyperparameter_space(full_dataset,
                                           num_estimators_length=HYPERPARAMETERS_CONFIG.num_estimators_length,
                                           lr_length=HYPERPARAMETERS_CONFIG.lr_length,
                                           max_depth_length=HYPERPARAMETERS_CONFIG.max_depth_length,
                                           seed=RND_STATE_SEED)

    info =  dotdict(
        dict(df_stats=pd.DataFrame({"oi": [11]}),
             feature_set=["a"],
             target="target",
             contains_categorical=False,
             categorical_features=None
             )
        )
    train_set, test_set = dataset_split(full_dataset,
                                        feature_set=info.feature_set,
                                        target=info.target,
                                        train_percentage=DATASET_PARAMETERS.train_percentage,
                                        seed=RND_STATE_SEED)
    
    binary_model_experiment(
        dataset=dotdict(dict(train_set=train_set, test_set=test_set)),
        hyperparameter_tree=hp_space,
        analyzer_info=info,
        prediction_column="prediction",
        eval_fn=auc_evaluator(
            target_column="target", prediction_column="prediction"
        )
    )
if __name__ == "__main__":
     run_binary_pipeline()
