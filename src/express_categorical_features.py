#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import pandas as pd
from ploting_devices import Ploting_devices as pld
import lib_kit.utils as lku

genres = "news,reviews,tales".split(",")
categoricals = """predominant_pers_pron,predominant_time""".split(",")


class CategoricalExpression:
    nice_names = {
        "predominant_pers_pron": {
            "name": "Personal Pronoun Prevalence",
            "new_indexes": "1st Person, 2nd Person, 3rd Person".split(",")
        },
        "predominant_time": {
            "name": "Verbal Time Prevalence",
            "new_indexes": ['Future Form', 'Past Form', 'Present Form']
        }
    }

    @staticmethod
    def categorical_grafics(dict_descriptive_dicts, category, path_graphs):
        """
           Receives the complete dictionary of features
           Generates a graph expressing the data
           :param dict_descriptive_dicts:
           :return:
           """

        titulo = CategoricalExpression.nice_names[category]["name"]
        nomFile = titulo.replace(" ", "_")
        graph_root = os.path.join(path_graphs, nomFile)

        val = []
        for genre in genres:
            val.append(dict_descriptive_dicts[genre]["df_categorical"][category]["normal_counts"])

        # Organizar para graficar
        # Le decimos que nos ponga las tres series como df y que sustituya los nan por 0
        normal_counts_df = pd.concat(val, axis=1).fillna(0).astype(float)
        # Renombrar columnas con generos
        normal_counts_df.columns = genres
        # Renombrar indices para la gráfica
        index_dic = {x: y for x, y in
                     zip(normal_counts_df.index, CategoricalExpression.nice_names[category]["new_indexes"])}
        normal_counts_df.rename(index_dic, axis='index', inplace=True)
        # Generar un diagrama representativo
        pld.stack_df(normal_counts_df.T, title=titulo, name_file=graph_root, format="eps")

    @staticmethod
    def categorical_printable_df(dict_descriptive_dicts, category):
        nice_view = lambda row: f'{row["value_counts"].astype(int)} ({row["normal_counts"]} %)'

        # Generar una tabla comparativa para cada categoria
        printable_dict_for_cat = {}
        for genre in genres:
            printable_dict_for_cat[genre] = dict_descriptive_dicts[genre]["df_categorical"][category].apply(nice_view,
                                                                                                            axis=1)
        printable_dict_df = pd.DataFrame(printable_dict_for_cat).fillna(0).astype(
            str)  # Renombrar indices para la gráfica
        index_dic = {x: y for x, y in
                     zip(printable_dict_df.index, CategoricalExpression.nice_names[category]["new_indexes"])}
        printable_dict_df.rename(index_dic, axis='index', inplace=True)

        return printable_dict_df

    @staticmethod
    def categorical_field_df(dict_descriptive_dicts, category, field):
        dict_for_cat_series = {}
        for genre in genres:
            dict_for_cat_series[genre] = dict_descriptive_dicts[genre]["df_categorical"][category][field]

        field_df = pd.DataFrame(dict_for_cat_series).fillna(0)  # Renombrar indices para la gráfica

        return field_df

    @staticmethod
    def categorical_latex(df, category, LATEX_PATH, feat_latex):
        titulo = CategoricalExpression.nice_names[category]["name"]
        nomFile = titulo.replace(" ", "_")
        latex_root = os.path.join(LATEX_PATH, nomFile)

        lku.dfs_to_latex(df, latex_root,
                         index=True,
                         label=feat_latex["label"],
                         caption=titulo,
                         header=feat_latex["header"],
                         decimal=feat_latex["decimal"],
                         column_format=feat_latex["column_format"],
                         float_format=feat_latex["float_format"]
                         )
