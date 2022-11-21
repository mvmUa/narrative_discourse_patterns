#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import pandas as pd
import configparser

import lib_kit.utils as lku
from ploting_devices import Ploting_devices as pld
from features_expression.express_categorical_features import CategoricalExpression as cex

LATEX_PATH = "latex_files/"

genres = "news,reviews,tales".split(",")


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

    dict_features_names["existence_proportions"] = config["DEFAULT"]["existence_proportions"].split(",")
    dict_features_names["elements_per_sentence"] = config["DEFAULT"]["elements_per_sentence"].split(",")
    dict_features_names["elements_per_group"] = config["DEFAULT"]["elements_per_group"].split(",")
    dict_features_names["other_comparison_doc_level"] = config["DEFAULT"]["other_comparison_doc_level"].split(",")
    dict_features_names["means"] = config["DEFAULT"]["means"].split(",")
    dict_features_names["categoricals"] = config["DEFAULT"]["categoricals"].split(",")

    return dict_features_names


def get_dtype_dict_for_dataset_corpus_type(dict_features_names):
    """
    A dictionary is generated with the name of the field and the corresponding data type.
    """
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


def read_genre_features_into_df(features_file_name, dict_features_names):
    """
    Reading of the file with the including the corresponding data types according to the feature.
    """
    df = pd.read_csv(features_file_name, sep=":")
    dict_type = get_dtype_dict_for_dataset_corpus_type(dict_features_names)

    for col in df.columns:
        df[col] = df[col].astype(dict_type[col])

    return df


def get_description_of_features(df, dict_features_names):
    """
    Receives the df of a genre
    Calculates the relevant statistics for 3 groups of characteristics_ countable, categorical, the remainder
    :param df:
    :return: dictionary 3 df
    """
    df_non_countables_described = df[[x for x in df.columns if x not in (
            dict_features_names["plain_counts"] + dict_features_names["existence_counts"])]].describe().T
    df_non_countables_described = df_non_countables_described["mean,std,min,max".split(",")]

    df_countables_described = df[
        dict_features_names["plain_counts"] + dict_features_names["existence_counts"]].describe().T
    df_countables_sum = df[dict_features_names["plain_counts"] + dict_features_names["existence_counts"]].sum(axis=0)
    df_countables_described["sum"] = df_countables_sum
    df_countables_described["docs_with_no_elems"] = (
            (df[dict_features_names["plain_counts"] + dict_features_names["existence_counts"]]) == 0).sum(axis=0)
    df_countables_described["prop_docs_with_no_elems"] = df_countables_described["docs_with_no_elems"] / len(df)
    df_countables_described = df_countables_described[
        "mean,std,sum,min,max,docs_with_no_elems,prop_docs_with_no_elems".split(",")]

    df_categorical_described_dict = {}
    # Personal Pronouns
    df_categorical_described_dict["predominant_pers_pron"] = pd.DataFrame(
        {"value_counts": df.predominant_pers_pron.value_counts()})
    df_categorical_described_dict["predominant_pers_pron"]["normal_counts"] = df.predominant_pers_pron.value_counts(
        normalize=True).mul(100).round(1)

    # Verbal
    df_categorical_described_dict["predominant_time"] = pd.DataFrame(
        {"value_counts": df.predominant_time.value_counts()})
    df_categorical_described_dict["predominant_time"]["normal_counts"] = df.predominant_time.value_counts(
        normalize=True).mul(100).round(1)

    tipes_of_feature_analysis = "df_countables,df_non_countables,df_categorical".split(",")
    tipes_of_df_described = [df_countables_described, df_non_countables_described, df_categorical_described_dict]
    dict_features_genre = {tipe: df_desc for tipe, df_desc in zip(tipes_of_feature_analysis, tipes_of_df_described)}

    return dict_features_genre


#######################################################
##########       Caracteristicas especiales
#######################################################

def analizar_sub_obj(dict_descriptive_dicts):
    # c_nsubj, c_csubj, c_xsubj, c_dobj
    feature_main_name = "subj_obj"

    features_to_fetch = "c_nsubj,c_csubj,c_xsubj,c_dobj".split(",")
    statistics_to_fetch = "mean,sum".split(",")
    df_subtype = "df_countables"

    # We obtain a dataframe for each statistic.
    feature_type_df_dict = get_dict_of_subdataframes(dict_descriptive_dicts, genres, df_subtype, feature_main_name,
                                                     features_to_fetch, statistics_to_fetch)

    # generate latex for each table
    for stat in statistics_to_fetch:
        # Pass a list of df's
        name_of_df = f"{feature_main_name}_{stat}"
        latex_root = os.path.join(LATEX_PATH, feature_main_name, name_of_df)
        create_dir(os.path.join(LATEX_PATH, feature_main_name))
        caption = {
            "sum": f'{feature_main_name.replace("_", " and ").capitalize()}. Estatistic: {stat} . Suma de todos los elementos del corpus.',
            "mean": f'{feature_main_name.replace("_", " and ").capitalize()}. Estatistic: {stat} . Media de cada característica por documento.'
        }
        lku.dfs_to_latex(df=feature_type_df_dict[name_of_df],

                         latexFileName=latex_root,
                         index=True,
                         label=f"tab:{name_of_df}",
                         caption=caption[stat],
                         header=list(feature_type_df_dict[name_of_df].columns),
                         decimal=".",
                         column_format="lccc"
                         )

    # Save table to excel
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(feature_type_df_dict, path_excel_file)


def analizar_timex(dict_descriptive_dicts):
    # existence_proportions =  prop_E_timex
    # elements_per_sentence = timexXsent --> words per sentence
    # existence_counts_readable = E_c_timexes,  E_DATE, E_TIME, E_DURATION, E_SET,
    # plain_counts_readable = c_timexes, DATE, TIME, DURATION, SET,
    feature_main_name = "timex"

    features_to_fetch_standard_stats = {
        "c_timexes_sum": {"feats": "c_timexes".split(","), "stat": "sum", "caption": "Total de timex por corpus"},
        "c_timexes_mean": {"feats": "c_timexes".split(","), "stat": "mean", "caption": "Media de timex por documento"},
        "c_timexes_docs_with_no_elems": {"feats": "c_timexes".split(","), "stat": "docs_with_no_elems",
                                         "caption": "Cantidad de documentos sin timex"},
        "c_timexes_prop_docs_with_no_elems": {"feats": "c_timexes".split(","), "stat": "prop_docs_with_no_elems",
                                              "caption": "Proporción de documentos sin timex"},
        "timexXsent": {"feats": "timexXsent".split(","), "stat": "mean",
                       "caption": "Media de timex por oracion : Promedio "},
        "E_c_timexes": {"feats": "E_c_timexes".split(","), "stat": "mean",
                        "caption": "Numero de oraciones del documento que contienen un timex : Promedio "},
        "prop_E_timex": {"feats": "prop_E_timex".split(","), "stat": "mean",
                         "caption": "Proporcion de oraciones en documento que contienen un timex : Promedio "}
    }

    # En un caso queremos evaluar la clasificacion dentro de la feature, Date, set, etc.
    # DATE, TIME, DURATION, SET,
    # E_DATE, E_TIME, E_DURATION, E_SET,
    # Otras preguntas: qué porcentaje de documentos no contiene ningun timex
    other_features_to_fetch = {
        "class_timex_docs_with_no_elems": {"feats": "DATE,TIME,DURATION,SET".split(","), "stat": "docs_with_no_elems",
                                           "caption": "Cantidad de documentos sin ELEMS"},
        "class_timex_prop_docs_with_no_elems": {"feats": "DATE,TIME,DURATION,SET".split(","),
                                                "stat": "prop_docs_with_no_elems",
                                                "caption": "Proporción de documentos sin ELEMS"},
        "class_E_timex_sum": {"feats": "E_DATE,E_TIME,E_DURATION,E_SET".split(","), "stat": "sum",
                              "caption": "Número de oraciones con un elemento ELEM de cada tipo,  por corpus"},
        "class_E_timex_mean": {"feats": "E_DATE,E_TIME,E_DURATION,E_SET".split(","), "stat": "mean",
                               "caption": "Media de número de oraciones con un elemento ELEM de cada tipo, por docu, por corpus"},
        "class_timex_sum": {"feats": "DATE,TIME,DURATION,SET".split(","), "stat": "sum",
                            "caption": "Cuantos timex de cada tipo hay por  corpus"},
        "class_timex_mean": {"feats": "DATE,TIME,DURATION,SET".split(","), "stat": "mean",
                             "caption": "Media timex de cada tipo hay por documento corpus"},
        "class_timex_distribution": {"feats": "DATE,TIME,DURATION,SET".split(","), "stat": "sum",
                                     "caption": "Distribución de tipos de timex por corpus"}
    }

    dict_dfs_to_excel = {}
    all_features_to_fetch = dict(features_to_fetch_standard_stats, **other_features_to_fetch)

    dict_dfs_to_excel["info"] = pd.DataFrame(all_features_to_fetch).T

    for feat_name, valores in all_features_to_fetch.items():

        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]

        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        if "distribution" not in feat_name:
            # We obtain a dataframe for each statistic.
            # df_gral, genres, sub_df_cat, feature_name, features_to_fetch, statistics_to_fetch
            dict_dfs_to_excel[feat_name] = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                                           features_to_fetch,
                                                                           statistics_to_fetch)
        else:
            aux_df = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                     features_to_fetch,
                                                     statistics_to_fetch)

            sums = {x: y for x, y in zip(list(aux_df.columns), aux_df.sum(axis=0).tolist())}
            aux_dic = {}
            for col in aux_df.columns:
                aux_dic[col] = aux_df[col].map(lambda x: (x / sums[col]) * 100)
            dict_dfs_to_excel[feat_name] = pd.DataFrame(aux_dic)

            # PLOT
            path_graphic_file = os.path.join(GRAPHS_PATH, feat_name)
            pld.stack_df(dict_dfs_to_excel[feat_name].T, title=valores["caption"], name_file=path_graphic_file,
                         format="eps")

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, all_features_to_fetch, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_categoricals(dict_descriptive_dicts):
    # Analysis of categorical features
    feature_main_name = "categoricals"
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)

    cat_dfs_dict = {}
    for category in dict_features_names["categoricals"]:
        # create graaphic for  normalized counts
        cex.categorical_grafics(dict_descriptive_dicts, category, GRAPHS_PATH)

        # generate a comparison table for each value-column: value_count and normal_counts
        fields = dict_descriptive_dicts["news"]["df_categorical"][category].columns
        for field in fields:
            cat_dfs_dict[category + "_" + field] = cex.categorical_field_df(dict_descriptive_dicts, category, field)

        # generate the df
        cat_dfs_dict[category + "_print"] = cex.categorical_printable_df(dict_descriptive_dicts, category)
        feat_latex = {
            "index": True,
            "label": f"tab:{category}",
            "caption": category.replace("_", " "),
            "header": genres,
            "decimal": ".",
            "column_format": "lccc",
            "float_format": "%.2f"
        }
        cex.categorical_latex(cat_dfs_dict[category + "_print"], category, LATEX_PATH, feat_latex)

    # save in excel file
    lku.dfs_to_excel_sheets(cat_dfs_dict.values(), cat_dfs_dict.keys(), path_excel_file, index=True)


def analizar_words_and_sentences(dict_descriptive_dicts):
    feature_main_name = "words_and_sentences"

    dict_dfs_to_excel = {}
    features = "words,c_sentences".split(",")
    for feat in features:
        df_aux = {}
        for genre in genres:
            df_aux[genre] = dict_descriptive_dicts[genre]["df_countables"].loc[feat]

        dict_dfs_to_excel[feat] = pd.DataFrame(df_aux)

        latex_root = os.path.join(LATEX_PATH, feature_main_name, feat)
        create_dir(os.path.join(LATEX_PATH, feature_main_name))
        lku.dfs_to_latex(dict_dfs_to_excel[feat],
                         latex_root,
                         index=True,
                         label=f"tab:{feat}_statistics",
                         caption=f"{feat} statistics".capitalize(),
                         header=list(dict_dfs_to_excel[feat].columns),
                         decimal=".",
                         column_format="lccc"
                         )

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_events(dict_descriptive_dicts):
    # Relacionados con eventos tenemos en df_countable, non_countable
    # countable: c_events, ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE
    #           E_c_events, E_ASPECTUAL,E_I_ACTION,E_I_STATE,E_OCCURRENCE,E_PERCEPTION,E_REPORTING,E_STATE,
    # existence_proportions = prop_E_event,  prop_E_occu, prop_E_perc, prop_E_repo,
    # elements_per_sentence =  eventsXsent,
    # elements_per_group = occuXtot_even, percXtot_even, repoXtot_even

    feature_main_name = "events"

    features_to_fetch_standard_stats = {
        "c_events_sum": {"feats": ["c_events"], "stat": "sum", "caption": "Total de events por corpus"},
        "c_events_mean": {"feats": ["c_events"], "stat": "mean", "caption": "Media de events por documento"},
        "c_events_docs_with_no_elems": {"feats": ["c_events"], "stat": "docs_with_no_elems",
                                        "caption": "Cantidad de documentos sin events"},
        "c_events_prop_docs_with_no_elems": {"feats": ["c_events"], "stat": "prop_docs_with_no_elems",
                                             "caption": "Proporción de documentos sin events"},
        "eventsXsent_mean": {"feats": ["eventsXsent"], "stat": "mean",
                             "caption": "Media de events por oracion : Promedio "},
        "E_c_events_mean": {"feats": ["E_c_events"], "stat": "mean",
                            "caption": "Numero de oraciones del documento que contienen un events : Promedio "},
        "prop_E_event_mean": {"feats": ["prop_E_event"], "stat": "mean",
                              "caption": "Proporcion de oraciones en documento que contienen un events : Promedio "}
    }

    # En un caso queremos evaluar la clasificacion dentro de la feature, Date, set, etc.
    # ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE
    # E_ASPECTUAL, ... I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE
    other_features_to_fetch = {
        "class_events_docs_with_no_elems": {
            "feats": "ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE".split(","),
            "stat": "docs_with_no_elems",
            "caption": "Cantidad de documentos sin ELEMS"},
        "class_events_prop_docs_with_no_elems": {
            "feats": "ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE".split(","),
            "stat": "prop_docs_with_no_elems",
            "caption": "Proporción de documentos sin ELEMS"},
        "class_E_events_count": {
            "feats": "E_ASPECTUAL,E_I_ACTION,E_I_STATE,E_OCCURRENCE,E_PERCEPTION,E_REPORTING,E_STATE".split(","),
            "stat": "sum",
            "caption": "Media de número de oraciones con un elemento ELEM de cada tipo, por docu, por corpus"},
        "class_events_count": {"feats": "ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE".split(","),
                               "stat": "sum",
                               "caption": "Cuantos events de cada tipo hay por corpus"},
        "class_events_distribution": {
            "feats": "ASPECTUAL,I_ACTION,I_STATE,OCCURRENCE,PERCEPTION,REPORTING,STATE".split(","), "stat": "sum",
            "caption": "Distribución de tipos de events por corpus"}
    }

    dict_dfs_to_excel = {}
    all_features_to_fetch = dict(features_to_fetch_standard_stats, **other_features_to_fetch)
    dict_dfs_to_excel["info"] = pd.DataFrame(all_features_to_fetch).T

    for feat_name, valores in features_to_fetch_standard_stats.items():
        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]

        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        #  We obtain a dictionary with a dataframe for each statistic.
        dict_dfs_to_excel[feat_name] = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                                       features_to_fetch,
                                                                       statistics_to_fetch)

    for feat_name, valores in other_features_to_fetch.items():
        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]

        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        if "distribution" not in feat_name:
            #  We obtain a dictionary with a dataframe for each statistic.
            # df_gral, genres, sub_df_cat, feature_name, features_to_fetch, statistics_to_fetch
            dict_dfs_to_excel[feat_name] = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                                           features_to_fetch,
                                                                           statistics_to_fetch)

        else:
            aux_df = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                     features_to_fetch,
                                                     statistics_to_fetch)

            sums = {x: y for x, y in zip(list(aux_df.columns), aux_df.sum(axis=0).tolist())}
            aux_dic_all = {}
            for col in aux_df.columns:
                aux_dic_all[col] = aux_df[col].map(lambda x: (x / sums[col]) * 100)
            dict_dfs_to_excel[feat_name] = pd.DataFrame(aux_dic_all)

            aux_df2 = dict_dfs_to_excel[feat_name].loc[[x for x in features_to_fetch if x != "OCCURRENCE"]]
            sums = {x: y for x, y in zip(list(aux_df2.columns), aux_df2.sum(axis=0).tolist())}
            aux_dic_all = {}
            for col in aux_df2.columns:
                aux_dic_all[col] = aux_df2[col].map(lambda x: (x / sums[col]) * 100)
            aux_df2 = pd.DataFrame(aux_dic_all)

            # GRAFICA 1: todos los tipos
            path_graphic_file = os.path.join(GRAPHS_PATH, feat_name)
            pld.stack_df(dict_dfs_to_excel[feat_name].T, title=valores["caption"],
                         name_file=path_graphic_file, format="eps")

            # GRAFICA 2: we remove occurence because it does not allow to observe the distribution of the rest of the types
            path_graphic_file = os.path.join(GRAPHS_PATH, feat_name + "_extracting_occurrence")
            pld.stack_df(aux_df2.T, title=valores["caption"] + "(drop occurrence)",
                         name_file=path_graphic_file, format="eps")

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, all_features_to_fetch, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_tlinks(dict_descriptive_dicts):
    # c_tlinks,
    # BEFORE,AFTER,IBEFORE,IAFTER,INCLUDES,IS_INCLUDED,BEGINS,BEGUN_BY,ENDS,ENDED_BY,SIMULTANEOUS,NONE,VAGUE,UNKNOWN,OVERLAP,BEFORE_OR_OVERLAP,OVERLAP_OR_AFTER,v
    feature_main_name = "tlinks"

    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats("tlinks", "c_tlinks", "tlinks")

    feats_to_fetch = "BEFORE,AFTER,IBEFORE,IAFTER,INCLUDES,IS_INCLUDED,BEGINS,BEGUN_BY,ENDS,ENDED_BY," \
                     "SIMULTANEOUS,NONE,VAGUE,UNKNOWN,OVERLAP,BEFORE_OR_OVERLAP,OVERLAP_OR_AFTER"
    # A traves de los calculos hemos comprobado que las relaciones temporales que NO son distintas de 0 son:
    feats_to_fetch = "BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,VAGUE"
    other_features_to_fetch = {
        "class_tlinks_docs_with_no_elems": {"feats": feats_to_fetch, "stat": "docs_with_no_elems",
                                            "caption": "Cantidad de documentos sin ELEMS"},
        "class_tlinks_prop_docs_with_no_elems": {"feats": feats_to_fetch, "stat": "prop_docs_with_no_elems",
                                                 "caption": "Proporción de documentos sin ELEMS"},
        "class_tlinks_count_sum": {"feats": feats_to_fetch, "stat": "sum",
                                   "caption": "Media de número de oraciones con un elemento ELEM de cada tipo, por docu, por corpus"},
        "class_tlinks_count_mean": {"feats": feats_to_fetch, "stat": "mean",
                                    "caption": "Cuantos timex de cada tipo hay por corpus"},
        "class_tlinks_distribution": {"feats": feats_to_fetch, "stat": "sum",
                                      "caption": "Distribución de tipos de timex por corpus"}
    }

    features_to_fetch_all = dict(features_to_fetch_standard_stats, **other_features_to_fetch)

    dict_dfs_to_excel = {}
    dict_dfs_to_excel["info"] = pd.DataFrame(dict(features_to_fetch_standard_stats, **other_features_to_fetch)).T

    for feat_name, valores in features_to_fetch_all.items():
        statistics_to_fetch = valores["stat"].split(",")
        f_to_fetch = valores["feats"].split(",")
        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (f_to_fetch[0] in y.index)][0]

        aux_df = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype, f_to_fetch,
                                                 statistics_to_fetch)

        if "distribution" not in feat_name:
            dict_dfs_to_excel[feat_name] = aux_df

        else:
            sums = {x: y for x, y in zip(list(aux_df.columns), aux_df.sum(axis=0).tolist())}
            aux_dic = {}
            for col in aux_df.columns:
                aux_dic[col] = aux_df[col].map(lambda x: (x / sums[col]) * 100)
            dict_dfs_to_excel[feat_name] = pd.DataFrame(aux_dic)

            # GRAFICA
            path_graphic_file = os.path.join(GRAPHS_PATH, feat_name + "_stacked")
            pld.stack_df(dict_dfs_to_excel[feat_name].T, title=valores["caption"], name_file=path_graphic_file,
                         format="eps")

            # GRAFICA BARRAS
            graph_path = os.path.join(GRAPHS_PATH, feat_name + "_bars")

            pld.grouped_barplot(dict_dfs_to_excel[feat_name], graph_path, "eps")

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_all, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_phrases_gram(dict_descriptive_dicts):
    # c_NP,c_NNP,c_VP,
    # E_c_NP,E_c_NNP,E_c_VP,
    # NPXsent, VPXsent, NNPXsent,
    feature_main_name = "phrases_gram"

    features_to_fetch_standard_stats = {
        "c_gram_phrases_sum": {"feats": "c_NP,c_NNP,c_VP".split(","), "stat": "sum",
                               "caption": "Total of different grammatical phrases per corpus"},
        "c_gram_phrases_mean": {"feats": "c_NP,c_NNP,c_VP".split(","), "stat": "mean",
                                "caption": "Mean of different grammatical phrases per document"},
        "c_gram_phrases_docs_with_no_elems": {"feats": "c_NP,c_NNP,c_VP".split(","), "stat": "docs_with_no_elems",
                                              "caption": "Proporción de documentos sin phrase"},
        "c_gram_phrases_prop_docs_with_no_elems": {"feats": "c_NP,c_NNP,c_VP".split(","),
                                                   "stat": "prop_docs_with_no_elems",
                                                   "caption": "Proporción de documentos sin phrase"},
        "E_gram_phrases_sum": {"feats": "E_c_NP,E_c_NNP,E_c_VP".split(","), "stat": "sum",
                               "caption": "Total sentences with grammatical phrases per corpus"},
        "E_gram_phrases_mean": {"feats": "E_c_NP,E_c_NNP,E_c_VP".split(","), "stat": "mean",
                                "caption": "Average number of sentences with grammatical phrases per document"},
        "gram_phrases_per_sent_mean": {"feats": "NPXsent,VPXsent,NNPXsent".split(","), "stat": "mean",
                                       "caption": "Average number of gram phrases per sentence"}
    }

    dict_dfs_to_excel = {}
    dict_dfs_to_excel["info"] = pd.DataFrame(features_to_fetch_standard_stats).T

    for feat, valores in features_to_fetch_standard_stats.items():
        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]
        new_dict_id = feat

        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        dict_dfs_to_excel[new_dict_id] = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype,
                                                                         features_to_fetch, statistics_to_fetch)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_sem_sentences(dict_descriptive_dicts):
    # exclamativas, interrogativas, parentesis
    # sent_exclamative, sent_exclamative_prop,
    # sent_interrogative, sent_interrogative_prop,
    # sent_parenthesis, sent_parenthesis_prop

    feature_main_name = "sem_sentences"

    feats = "sent_exclamative,sent_interrogative,sent_parenthesis".split(",")
    features_to_fetch_standard_stats = {
        f"{feature_main_name}_sum": {"feats": feats, "stat": "sum",
                                     "caption": f"Total of semantic sentences in set (exc,interr,parenth) per corpus"},
        f"{feature_main_name}_mean": {"feats": feats, "stat": "mean",
                                      "caption": f"Mean of semantic sentences in set (exc,interr,parenth) per document"},
        f"{feature_main_name}_prop": {"feats": "sent_exclamative_prop,sent_interrogative_prop".split(","),
                                      "stat": "mean",
                                      "caption": f"Mean of semantic sentences in set (exc,interr,parenth) per document"},
        f"{feature_main_name}_docs_with_no_elems": {"feats": feats, "stat": "docs_with_no_elems",
                                                    "caption": f"Num de documentos sin semantic sentences"},
        f"{feature_main_name}_prop_docs_with_no_elems": {"feats": feats, "stat": "prop_docs_with_no_elems",
                                                         "caption": f"Proporción de documentos sin sem sentence of class"}
    }

    dict_dfs_to_excel = {}
    dict_dfs_to_excel["info"] = pd.DataFrame(features_to_fetch_standard_stats).T

    for feat_name, valores in features_to_fetch_standard_stats.items():
        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]

        df_subtype = [x for x, y in dict_descriptive_dicts["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        aux = get_dataframe_by_feats_and_stat(dict_descriptive_dicts, genres, df_subtype, features_to_fetch,
                                              statistics_to_fetch)

        dict_dfs_to_excel[feat_name] = aux

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_participle_clauses(dict_descriptive_dicts):
    # participle_be, participle_have
    feature_main_name = "participle_clauses"
    feats_to_fetch = "participle_be,participle_have".split(",")
    feats_string = "participle clauses"

    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                            feats_string)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_personal_pronouns_analysis(dict_descriptive_dicts):
    #         # pers_pronouns_1P,pers_pronouns_2P,pers_pronouns_3P
    #         # E_1P,E_2P
    #         # E_1P_prop,E_2P_prop
    #         # noun_proper,noun_proper_uniqs,NP_noun,E_NP,E_NP_prop
    feature_main_name_gral = "personal_pronouns"

    groups_of_features = "pers_pronouns_1P,pers_pronouns_2P,pers_pronouns_3P:E_1P,E_2P:E_1P_prop,E_2P_prop:noun_proper,noun_proper_uniq,E_NP:NP_nouns,E_NP_prop".split(
        ":")
    groups_name = "plain,existence,prop_existence,proper_nouns_plain,proper_nouns_prop".split(",")
    types_features = "plain,existence,prop,plain,prop".split(",")

    for group_name, type_feature, feats_to_fetch in zip(groups_name, types_features, groups_of_features):
        feature_main_name = feature_main_name_gral + "_" + group_name
        print(feature_main_name)

        feats_to_fetch = feats_to_fetch.split(",")
        feats_string = "ELEM"

        features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                                feats_string, type_feature)

        dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

        # We save the tables in latex
        dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

        #  We save the tables in an excel file
        path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
        dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_pronoun_analysis(dict_descriptive_dicts):
    # pers_pronouns_IT,
    # wh-pronoun

    feature_main_name = "pronoun_it_wh"
    feats_to_fetch = "pers_pronouns_IT,wh-pronoun".split(",")
    feats_string = "pronoun it and wh-"

    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                            feats_string)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_NER_analysis(dict_descriptive_dicts):
    # ner,E_NER,E_NER_prop,NER_NP,
    # B-LOC,B-MISC,B-ORG,B-PER,
    # NER_length_1,NER_length_2,NER_length_3,NER_length_4,NER_length_5_or_more

    feature_main_name = "NER"
    feats_to_fetch = "NER,E_NER,E_NER_prop,NER_NP".split(",")  # 1 1-2 3
    feats_string = "NER"
    feats_for_classes = "B-LOC,B-MISC,B-ORG,B-PER".split(",")

    features_to_fetch_standard_stats = {
        "NER_sum": {"feats": "NER", "stat": "sum", "caption": "Total de NER por corpus"},
        "NER_mean": {"feats": "NER", "stat": "mean", "caption": "Media de NER por documento"},
        "NER_docs_with_no_elems": {"feats": "NER", "stat": "docs_with_no_elems",
                                   "caption": "Cantidad de documentos sin NER"},
        "NER_prop_docs_with_no_elems": {"feats": "NER", "stat": "prop_docs_with_no_elems",
                                        "caption": "Proporción de documentos sin NER"},
        "NER_NP": {"feats": "NER_NP".split(","), "stat": "mean", "caption": "Media de NER por oracion : Promedio "},
        "E_NER": {"feats": "E_NER".split(","), "stat": "mean",
                  "caption": "Numero de oraciones del documento que contienen un NER : Promedio "},
        "prop_E_NER": {"feats": "prop_E_NER".split(","), "stat": "mean",
                       "caption": "Proporcion de oraciones en documento que contienen un NER : Promedio "}
    }

    other_features_to_fetch = {
        "class_NER_docs_with_no_elems": {"feats": feats_for_classes, "stat": "docs_with_no_elems",
                                         "caption": "Cantidad de documentos sin ELEMS"},
        "class_NER_prop_docs_with_no_elems": {"feats": feats_for_classes, "stat": "prop_docs_with_no_elems",
                                              "caption": "Proporción de documentos sin ELEMS"},
        "class_E_NER_sum": {"feats": "E_DATE,E_TIME,E_DURATION,E_SET".split(","), "stat": "sum",
                            "caption": "Número de oraciones con un elemento ELEM de cada tipo,  por corpus"},
        "class_E_NER_mean": {"feats": "E_DATE,E_TIME,E_DURATION,E_SET".split(","), "stat": "mean",
                             "caption": "Media de número de oraciones con un elemento ELEM de cada tipo, por docu, por corpus"},
        "class_NER_sum": {"feats": feats_for_classes, "stat": "sum",
                          "caption": "Cuantos NER de cada tipo hay por  corpus"},
        "class_NER_mean": {"feats": feats_for_classes, "stat": "mean",
                           "caption": "Media NER de cada tipo hay por documento corpus"},
        "class_NER_distribution": {"feats": feats_for_classes, "stat": "sum",
                                   "caption": "Distribución de tipos de NER por corpus"}
    }
    features_to_fetch_standard_stats = {**features_to_fetch_standard_stats,
                                        **other_features_to_fetch}

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_nouns_adj_adv_analysis(dict_descriptive_dicts):
    #     nouns,
    #     synsetAdv, wh - adverb, , when - adverb,
    #     synsetAdj, E_adj, E_adj_prop, adj_sent, adj_sust, adj_words,

    feature_main_name_gral = "nouns_adj_adv"

    groups_of_features = "nouns:synsetAdv,wh-adverb,when-adverb:synsetAdj:E_adj:E_adj_prop,adj_sent,adj_sust,adj_words".split(
        ":")
    groups_name = "nouns,adverbs,adjectives,Adj_existence,adjs_proportions".split(",")
    types_features = "plain,plain,plain,existence,prop".split(",")

    for group_name, type_feature, feats_to_fetch in zip(groups_name, types_features, groups_of_features):
        feature_main_name = feature_main_name_gral + "_" + group_name
        print(feature_main_name)

        feats_to_fetch = feats_to_fetch.split(",")
        feats_string = "ELEM"

        features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                                feats_string, type_feature)

        dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

        # We save the tables in latex
        dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

        #  We save the tables in an excel file
        path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
        dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_predicative_analysis(dict_descriptive_dicts):
    # predicative complement (atributo) implica una estructura X is Y "the pan was gorgeous" "we were impressed"
    # counts["predicative_complement"] = len(dataLine[dataLine.dep_tag.str.contains("PRD")])

    feature_main_name = "predicative_complement"
    feats_to_fetch = "predicative_complement".split(",")
    feats_string = "predicative_complement"

    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                            feats_string)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_quotation_analysis(dict_descriptive_dicts):
    # # quotation_marks, E_quotation_marks,

    feature_main_name = "quotation_marks"
    feats_to_fetch = "quotation_marks,E_quotation_marks".split(",")
    feats_string = "quotation marks"

    # recuperamos las medidas standard
    features_to_fetch_standard_stats_plain = get_features_to_fetch_standard_stats(feature_main_name,
                                                                                  [feats_to_fetch[0]], feats_string)
    features_to_fetch_standard_stats_existence = get_features_to_fetch_standard_stats("E_" + feature_main_name,
                                                                                      [feats_to_fetch[1]], feats_string,
                                                                                      "existence")
    features_to_fetch_standard_stats = dict(features_to_fetch_standard_stats_plain,
                                            **features_to_fetch_standard_stats_existence)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_word_length_analysis(dict_descriptive_dicts):
    # word_length_1,word_length_2,word_length_3,word_length_4,word_length_5,word_length_6,word_length_min_7,

    feature_main_name = "participle_clauses"
    feats_to_fetch = "participle_be,participle_have".split(",")
    feats_string = "participle clauses"

    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                            feats_string)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_corref_analysis(dict_descriptive_dicts):
    # chain_amount,mean_chain_len,chain_spread_mean,maximal_len_chains_amount,entity_concentration_mean,

    feature_main_name = "corref"
    feats_to_fetch = "chain_amount,maximal_len_chains_amount,mean_chain_len,chain_spread_mean,entity_concentration_mean".split(
        ",")  #
    feats_string = "corref ELEM"

    a = get_features_to_fetch_standard_stats(feature_main_name + "_chain", feats_to_fetch[0:1], feats_string)
    b = get_features_to_fetch_standard_stats(feature_main_name + "_maximal_chains", feats_to_fetch[1:2], feats_string)
    c = get_features_to_fetch_standard_stats(feature_main_name + "_prop", feats_to_fetch[2:], feats_string, "prop")

    features_to_fetch_standard_stats = {**a, **b, **c}

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_figures_analysis(dict_descriptive_dicts):
    # figures,E_Numbers,E_Numbers_prop,numbers_words,

    feature_main_name = "figures"
    feats_to_fetch = "figures,E_Numbers,E_Numbers_prop,numbers_words".split(",")  # 1 1-2 3
    feats_string = "figures"

    features_to_fetch_standard_stats_plain = get_features_to_fetch_standard_stats(feature_main_name,
                                                                                  feats_to_fetch[0:1], feats_string)
    features_to_fetch_standard_stats_existence = get_features_to_fetch_standard_stats("E_" + feature_main_name,
                                                                                      feats_to_fetch[1:2], feats_string,
                                                                                      "existence")
    features_to_fetch_standard_stats_prop = get_features_to_fetch_standard_stats(feature_main_name + "_prop",
                                                                                 feats_to_fetch[2:], feats_string,
                                                                                 "prop")
    features_to_fetch_standard_stats = {**features_to_fetch_standard_stats_plain,
                                        **features_to_fetch_standard_stats_existence,
                                        **features_to_fetch_standard_stats_prop}

    # guardamos en excel
    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_verb_analysis(dict_descriptive_dicts):
    # vform_future,vform_past,vform_present,vform_present_3erd,
    # predominant_time,
    # vform_gerund,vform_infinitive,vform_participle
    # verb_modal

    feature_main_name = "verbs_analysis"
    feats_to_fetch = "vform_future,vform_past,vform_present,vform_present_3erd,predominant_time,vform_gerund,vform_infinitive,vform_participle,verb_modal".split(
        ",")
    feats_string = "verb ELEM"

    a = get_features_to_fetch_standard_stats(feature_main_name + "_time", feats_to_fetch[0:3], feats_string)
    b = get_features_to_fetch_standard_stats(feature_main_name + "_present3erd", feats_to_fetch[3:4], feats_string)
    c = get_features_to_fetch_standard_stats(feature_main_name + "_form", feats_to_fetch[5:8], feats_string)
    d = get_features_to_fetch_standard_stats(feature_main_name + "_modals", feats_to_fetch[8:], feats_string)

    features_to_fetch_standard_stats = {**a, **b, **c, **d}

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_DMs_analysis(dict_descriptive_dicts):
    # DiscM,

    feature_main_name = "DiscM_general"
    feats_to_fetch = "DiscM".split(",")
    feats_string = "discourse markers (general)"

    # recuperamos las medidas standard
    features_to_fetch_standard_stats = get_features_to_fetch_standard_stats(feature_main_name, feats_to_fetch,
                                                                            feats_string)

    # guardamos en excel
    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_intr_Phr_analysis(dict_descriptive_dicts):
    # intr_ph,intr_with_PN_NER,intr_with_adverb,intr_ph_prop,

    feature_main_name = "figures"
    feats_to_fetch = "figures,E_Numbers,E_Numbers_prop,numbers_words".split(",")  # 1 1-2 3
    feats_string = "figures"

    features_to_fetch_standard_stats_plain = get_features_to_fetch_standard_stats(feature_main_name,
                                                                                  feats_to_fetch[0:1], feats_string)
    features_to_fetch_standard_stats_existence = get_features_to_fetch_standard_stats("E_" + feature_main_name,
                                                                                      feats_to_fetch[1:2], feats_string,
                                                                                      "existence")
    features_to_fetch_standard_stats_prop = get_features_to_fetch_standard_stats(feature_main_name + "_prop",
                                                                                 feats_to_fetch[2:], feats_string,
                                                                                 "prop")
    features_to_fetch_standard_stats = {**features_to_fetch_standard_stats_plain,
                                        **features_to_fetch_standard_stats_existence,
                                        **features_to_fetch_standard_stats_prop}

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


def analizar_commas_analysis(dict_descriptive_dicts):
    # c_commas, commas_prop

    feature_main_name = "commas"
    feats_to_fetch = "c_commas,commas_prop".split(",")
    feats_string = "commas"

    features_to_fetch_standard_stats_plain = get_features_to_fetch_standard_stats(feature_main_name,
                                                                                  [feats_to_fetch[0]], feats_string)
    features_to_fetch_standard_stats_existence = get_features_to_fetch_standard_stats(feature_main_name + "_prop",
                                                                                      [feats_to_fetch[1]], feats_string,
                                                                                      "prop")
    features_to_fetch_standard_stats = dict(features_to_fetch_standard_stats_plain,
                                            **features_to_fetch_standard_stats_existence)

    dict_dfs_to_excel = analizar_standard(dict_descriptive_dicts, features_to_fetch_standard_stats)

    # Arreglar un error que se cometió en freeling count
    dict_dfs_to_excel['commas_prop_mean'] = dict_dfs_to_excel['commas_prop_mean'] / 100

    # We save the tables in latex
    dict_of_df_to_latex(dict_dfs_to_excel, features_to_fetch_standard_stats, feature_main_name)

    #  We save the tables in an excel file
    path_excel_file = os.path.join(EXCEL_PATH, feature_main_name)
    dict_of_df_to_excel(dict_dfs_to_excel, path_excel_file)


###############################################
#     HERRAMIENTAS
##############################################

def get_features_to_fetch_standard_stats(feature_name, feats_to_fetch, feats_string, type_feature="plain"):
    features_to_fetch_standard_stats = {
        f"{feature_name}_sum": {"feats": feats_to_fetch, "stat": "sum",
                                "caption": f"Total of {feats_string} per corpus"},
        f"{feature_name}_mean": {"feats": feats_to_fetch, "stat": "mean",
                                 "caption": f"Mean of {feats_string} per document"},
        f"{feature_name}_docs_with_no_elems": {"feats": feats_to_fetch, "stat": "docs_with_no_elems",
                                               "caption": f"Num de documentos sin {feats_string}"},
        f"{feature_name}_prop_docs_with_no_elems": {"feats": feats_to_fetch, "stat": "prop_docs_with_no_elems",
                                                    "caption": f"Proporción de documentos sin {feats_string}"}
    }
    if (type_feature == "existence") or (type_feature == "prop"):
        del features_to_fetch_standard_stats[f"{feature_name}_docs_with_no_elems"]
        del features_to_fetch_standard_stats[f"{feature_name}_prop_docs_with_no_elems"]
    if (type_feature == "prop"):
        del features_to_fetch_standard_stats[f"{feature_name}_sum"]

    return features_to_fetch_standard_stats


def analizar_standard(dfgral, features_to_fetch_standard_stats):
    dict_dfs_to_excel = {}

    dict_dfs_to_excel["info"] = pd.DataFrame(features_to_fetch_standard_stats).T

    for feat_name, valores in features_to_fetch_standard_stats.items():
        statistics_to_fetch = [valores["stat"]]
        features_to_fetch = valores["feats"]

        df_subtype = [x for x, y in dfgral["news"].items() if
                      (isinstance(y, pd.core.frame.DataFrame)) and (features_to_fetch[0] in y.index)][0]

        aux = get_dataframe_by_feats_and_stat(dfgral, genres, df_subtype, features_to_fetch, statistics_to_fetch)

        dict_dfs_to_excel[feat_name] = aux

    return dict_dfs_to_excel


def dict_of_df_to_latex(dict_dfs_to_keep, features_to_fetch_standard_stats, feature_main_name):
    for feat_name, df_to_save in dict_dfs_to_keep.items():
        if feat_name != "info":
            latex_root = os.path.join(LATEX_PATH, feature_main_name, feat_name)
            create_dir(os.path.join(LATEX_PATH, feature_main_name))

            # Generamos un latex para cada tabla
            lku.dfs_to_latex(df=df_to_save,
                             latexFileName=latex_root,
                             index=True,
                             label=f"tab:{feat_name}",
                             caption=features_to_fetch_standard_stats[feat_name]["caption"] + f" - tab:{feat_name}",
                             header=list(df_to_save.columns),
                             decimal=".",
                             column_format="lccc"
                             )


def dict_of_df_to_excel(dict_dfs_to_keep, path_excel_file):
    #  We save the tables in an excel file
    lku.dfs_to_excel_sheets(dict_dfs_to_keep.values(), dict_dfs_to_keep.keys(), path_excel_file,
                            index=True)


def get_dict_of_subdataframes(df_gral, genres, sub_df_cat, feature_name, features_to_fetch, statistics_to_fetch):
    """
    There are n statistics that I need separately (x.e. sum and mean).
    There are m features that I am interested in (x.e. [nsubj,csubj,...] or [B-LOC, B-MISC,...]).
    A dataframe per statistic is required with those features that includes all 3 genres.
    This function returns a dictionary with one entry per statistic with the associated dataframe. Key: f"{feature_name}_{type_statistic}"
    :param df_gral:
    :param genres:
    :param statistics_to_fetch: "mean,sum".split(",") : array of stats
    :param feature_name:  e.g. "subj_and_objt"
    :param features_to_fetch:  e.g. "c_nsubj,c_csubj,c_xsubj,c_dobj".split(",")
    :return:
    """
    feature_type_df_dict = {}
    for type_statistic in statistics_to_fetch:
        aux_statistic = {}
        for genre in genres:
            fields_to_fetch = type_statistic
            aux_statistic[genre] = df_gral[genre][sub_df_cat].loc[features_to_fetch, fields_to_fetch]
        feature_type_df_dict[f"{feature_name}_{type_statistic}"] = pd.DataFrame(aux_statistic).fillna(0)

    return feature_type_df_dict


def get_dataframe_by_feats_and_stat(df_gral, genres, sub_df_cat, features_to_fetch, statistic_to_fetch):
    aux_statistic = {}
    for genre in genres:
        fields_to_fetch = statistic_to_fetch
        aux_statistic[genre] = df_gral[genre][sub_df_cat].loc[features_to_fetch, fields_to_fetch].T
        aux_statistic[genre].rename(index={statistic_to_fetch[0]: genre}, inplace=True)

    the_df = pd.concat(aux_statistic.values()).T

    return the_df


def create_dir(pathOut):
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)


def get_general_corpus_description(dict_descriptive_dicts):
    general = []
    for genre in genres:
        aux = {}
        aux["Genre"] = genre.capitalize()
        aux["num_docs"] = "completar"

        aux["total_sents"] = dict_descriptive_dicts[genre]["df_countables"].loc["c_sentences"]["sum"]
        aux["total_words"] = dict_descriptive_dicts[genre]["df_countables"].loc["words"]["sum"]

        aux["sents_per_doc"] = dict_descriptive_dicts[genre]["df_countables"].loc["c_sentences"]["mean"]
        aux["words_per_doc"] = dict_descriptive_dicts[genre]["df_countables"].loc["words"]["mean"]

        aux["wordsXsents_1"] = dict_descriptive_dicts[genre]["df_non_countables"].loc["wordsXsent"]["mean"]

        general.append(aux)

    df_to_save = pd.DataFrame(general).set_index("Genre")
    df_to_save.columns = "# docs,# sents,# words,sents/doc,words/doc,words/sent".split(",")

    latex_root = os.path.join(LATEX_PATH, "general_desc_stats.latex")
    # Generamos un latex para cada tabla
    lku.dfs_to_latex(df=df_to_save,
                     latexFileName=latex_root,
                     index=True, float_format="%.0f",
                     label=f"tab:generalStatsCorpora",
                     caption="Description of the corpora",
                     header=list(df_to_save.columns),
                     decimal=".",
                     column_format="lcccccc"
                     )
