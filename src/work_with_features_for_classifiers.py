#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import configparser


def read_config_features(config_file):
    dict_features_names = {}
    config = configparser.ConfigParser()
    config.read(config_file)

    dict_features_names["columns"] = config["DEFAULT"]["columnas_readable"].replace(" ", "").replace("\n", "").split(
        ",")
    dict_features_names["plain_counts"] = config["DEFAULT"]["plain_counts_readable"].replace(" ", "").replace("\n",
                                                                                                              "").split(
        ",")
    dict_features_names["existence_counts"] = config["DEFAULT"]["existence_counts_readable"].replace(" ", "").replace(
        "\n", "").split(",")

    dict_features_names["existence_proportions"] = config["DEFAULT"]["existence_proportions"].split(
        ",")
    dict_features_names["elements_per_sentence"] = config["DEFAULT"]["elements_per_sentence"].split(
        ",")
    dict_features_names["elements_per_group"] = config["DEFAULT"]["elements_per_group"].split(",")
    dict_features_names["other_comparison_doc_level"] = config["DEFAULT"]["other_comparison_doc_level"].split(",")
    dict_features_names["means"] = config["DEFAULT"]["means"].split(",")
    dict_features_names["categoricals"] = config["DEFAULT"]["categoricals"].split(",")

    return dict_features_names


def get_dtype_dict_for_dataset_corpus_type(dict_features_names):
    features_types_type = [int, int, float, float, float, float, float, "category"]
    features_types_names = "plain_counts,existence_counts,existence_proportions,elements_per_sentence,elements_per_group,other_comparison_doc_level,means,categoricals".split(
        ",")
    ini_dict = {x: y for x, y in zip(features_types_names, features_types_type)}
    types_dict = {}
    types_dict['file'] = str

    for type_feature_name, type_data in ini_dict.items():
        for item in dict_features_names[type_feature_name]:
            types_dict[item] = type_data
    return types_dict


def print_description_of_features(dict_features_names):
    features_types_type = [int, int, float, float, float, float, float, "category"]

    # cuantas features hay
    num_cols = len(dict_features_names['columns'])
    print(f"Hay un total de {num_cols}, pero una es el nom_file, entonces {num_cols - 1}")

    num_sumado = sum([len(x) for x in dict_features_names.values() if len(x) < 150])
    print(f"Para comprobar, sumamos el resto de características\nTotal:{num_sumado}")

    #      Imprimir listado
    # plain_counts: 93 features of type <class 'int'>; Examples:['c_sentences', 'c_events', 'ASPECTUAL']
    # existence_counts: 28 features of type <class 'int'>; Examples:['E_c_events', 'E_c_timexes', 'E_c_words']
    # existence_proportions: 14 features of type <class 'float'>; Examples:['prop_E_event', 'prop_E_timex', 'prop_E_occu']
    # elements_per_sentence: 8 features of type <class 'float'>; Examples:['wordsXsent', 'eventsXsent', 'timexXsent']
    # elements_per_group: 3 features of type <class 'float'>; Examples:['occuXtot_even', 'percXtot_even', 'repoXtot_even']
    # other_comparison_doc_level: 5 features of type <class 'float'>; Examples:['NER_NP', 'NP_nouns', 'numbers_words']
    # means: 3 features of type <class 'float'>; Examples:['mean_chain_len', 'chain_spread_mean', 'entity_concentration_mean']
    # categoricals: 2 features of type category; Examples:['predominant_pers_pron', 'predominant_time']
    for a, b, c in zip([x for x, y in dict_features_names.items() if x != "columns"],
                       [y for x, y in dict_features_names.items() if x != "columns"],
                       features_types_type):
        print(f"{a}: {len(b)} features of type {c}; Examples:{b[0:3]}")

    # Poner la informacion enun df
    df = pd.DataFrame()
    df["feature_class"] = [x for x, y in dict_features_names.items() if x != "columns"]
    features_sets = [y for x, y in dict_features_names.items() if x != "columns"]
    df["feature_len"] = [len(y) for y in features_sets]
    df["feature_type"] = features_types_type
    df["examples"] = [",".join(x[0:3]) for x in features_sets]

    return df


def dfs_to_excel_sheets(array_dfs, sheet_names, excelFileName, index=False):
    with pd.ExcelWriter(excelFileName + '.xlsx') as writer:
        for df, fname in zip(array_dfs, sheet_names):
            df.to_excel(writer, sheet_name=fname, index=index)
