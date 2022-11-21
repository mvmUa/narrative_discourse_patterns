#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import joblib as joblib
from functools import reduce
import os
import pandas as pd

main_root = "corpora_outs"


def subcorpus_join(subcorpus_name, genre, ml_type):
    print(subcorpus_name)
    df_list = []
    for tool in "free,caevo,coref".split(","):
        subcorpus_path = f"{main_root}/corpus_{ml_type}/{genre}/{tool}_{subcorpus_name}/{subcorpus_name}_{tool}_features.joblib"
        df_aux = joblib.load(subcorpus_path)
        df_aux = df_aux.drop_duplicates()
        df_aux = df_aux.dropna()
        print(len(df_aux))
        df_list.append(df_aux)
    print("%%%%%%%%%%")
    df_final = reduce(lambda left, right: pd.merge(left, right, on='file'), df_list)
    df_final = df_final.drop_duplicates()

    return df_final


def get_list_feats(subcorpus_name, genre, ml_type):
    dic_nom_features = {}

    print(subcorpus_name)
    df_list = []
    for tool in "free,caevo,coref".split(","):
        subcorpus_path = f"{main_root}/corpus_{ml_type}/{genre}/{tool}_{subcorpus_name}/{subcorpus_name}_{tool}_features.joblib"
        df_aux = joblib.load(subcorpus_path)
        for x in df_aux.columns:
            if x in dic_nom_features.keys():
                x = f"{x}__{tool[0:2]}"
            dic_nom_features.update({x: [tool[0:2]]})

    d_aux = pd.DataFrame.from_dict(dic_nom_features, orient="index")
    ex_file = "cols_and_tools.xlsx"
    d_aux.to_excel(ex_file)


def files_resultant_after_caevo_freeling_length_sentences_comparison(df_whole):
    comparable_df = df_whole[["file", "c_sentences", "f_sentences"]]
    comparable_df["dif"] = comparable_df.apply(lambda x: abs(x["c_sentences"] - x['f_sentences']), axis=1)
    comparable_df = comparable_df.sort_values(by="dif", ascending=True)

    comparable = comparable_df.iloc[0:2535]

    return comparable.file.tolist()


def saveandjointheData():
    # GENERA
    list_corpus_len = []
    df_whole_list = []
    for ml_type, corpora in corpora_types_genres.items():
        df_ml_type_list = []
        for genre, sub_corpora in corpora.items():
            df_genre_type_list = []
            for subcorpus_name in sub_corpora:
                subcorpus_joined = subcorpus_join(subcorpus_name, genre, ml_type)

                genre_path_csv = f"{main_root}/corpus_{ml_type}/{genre}/{subcorpus_name}.csv"
                subcorpus_joined.to_csv(index=False, path_or_buf=genre_path_csv)

                genre_path_job = f"{main_root}/corpus_{ml_type}/{genre}/{subcorpus_name}.joblib"
                joblib.dump(value=subcorpus_joined, filename=genre_path_job, compress="gzip")

                df_genre_type_list.append(subcorpus_joined)
                list_corpus_len.append([ml_type, genre, subcorpus_name, len(subcorpus_joined)])

            df_genre_type_joined = pd.concat(df_genre_type_list)
            df_genre_type_joined["genre"] = genre
            big_genre_path_csv = f"{main_root}/corpus_{ml_type}/{genre}/{genre}.csv"
            df_genre_type_joined.to_csv(index=False, path_or_buf=big_genre_path_csv)

            big_genre_path_csv_job = f"{main_root}/corpus_{ml_type}/{genre}/{genre}.joblib"
            joblib.dump(value=df_genre_type_joined, filename=big_genre_path_csv_job, compress="gzip")

            df_ml_type_list.append(df_genre_type_joined)

        df_ml_type = pd.concat(df_ml_type_list)
        big_ml_path_csv = f"{main_root}/corpus_{ml_type}/corpus_{ml_type}.csv"
        df_ml_type.to_csv(index=False, path_or_buf=big_ml_path_csv)

        big_ml_path_job = f"{main_root}/corpus_{ml_type}/corpus_{ml_type}.joblib"
        joblib.dump(value=df_ml_type, filename=big_ml_path_job, compress="gzip")

        df_whole_list.append(df_ml_type)

    df_whole_ini = pd.concat(df_whole_list)
    list_of_files = files_resultant_after_caevo_freeling_length_sentences_comparison(df_whole_ini)
    df_whole_max_dif_16_321_feats = df_whole_ini[df_whole_ini.file.isin(list_of_files)]

    whole_path_csv = f"{main_root}/feats_norm_and_genre.csv"
    df_whole_max_dif_16_321_feats.to_csv(index=False, path_or_buf=whole_path_csv)

    whole_path_job = f"{main_root}/feats_norm_and_genre.joblib"
    joblib.dump(value=df_whole_max_dif_16_321_feats, filename=whole_path_job, compress="gzip")

    df_lens = pd.DataFrame(list_corpus_len, columns="ml_type,genre,corpus,num_docs".split(","))
    df_lens.to_csv(index=False, path_or_buf=f"{main_root}/list_corpus_and_num_docs.csv")
    print(df_lens.groupby("ml_type").sum())
    print(df_lens.groupby("genre").sum())
    print(df_lens.groupby("corpus").sum())


def cureDataset():
    whole_path_job = f"{main_root}/feats_norm_and_genre.joblib"
    feat_raw_file = os.path.join(main_root, whole_path_job)
    feat_curated_file = os.path.join(main_root, whole_path_job)

    df = joblib.load(feat_raw_file)

    first_filtered_features = [x for x in df.columns if
                               x not in ['file', 'c_words_avg', 'E_words', 'E_words_prop', 'words_avg']]
    in_avg_feats = [x for x in first_filtered_features if "avg" in x]
    in_prop_feats = [x for x in first_filtered_features if ("prop" in x) and (not "proper" in x)]
    in_X_feats = [x for x in first_filtered_features if "X" in x]
    in_coref_feat_to_include = ['mean_chain_len', 'chain_spread_mean', 'entity_concentration_mean', 'chain_amount_avg',
                                'maximal_len_chains_amount_avg']
    in_other_free_ratios = ['adj_nouns',
                            'NP_nouns', 'NER_NP', 'predominant_time', 'predominant_pers_pron',
                            'predominant_NER_type', 'noun_proper_uniq']

    feats_interesting = set(in_avg_feats + \
                            in_prop_feats + \
                            in_X_feats + \
                            in_coref_feat_to_include + \
                            in_other_free_ratios + ["genre"])

    df_to_keep = df.loc[:, feats_interesting]
    joblib.dump(value=df_to_keep, filename=feat_curated_file, compress="gzip")
