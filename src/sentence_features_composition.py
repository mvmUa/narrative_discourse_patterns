#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd

ROOT_EXPERIMENTS = os.getcwd()
sent_features_gral_path = ROOT_EXPERIMENTS + "/integer/corpora_outs/sent_level_features"

tool_type = "caevo,coref,free".split(",")

corpora_genres_test = {
    "news": ["duc2002"],
    "tales": ["lym", "andersen"],
    "reviews": ["msd", "opin"]
}

corpora_genres_train = {
    "news": ["duc2004"],
    "tales": ["lym", "dormir"],
    "reviews": ["sfu"]
}

super_corpora = {
    "test": corpora_genres_test,
    "train": corpora_genres_train
}

special_codes = {x: y for x, y in
                 zip("COOKWARE,COMPUTERS,MOVIES,MOVIES2".split(","),
                     "CK,CP,MO,M2".split(","))}


def codify_file_name(dirName, fname, corpus_code):
    prefix = corpus_code
    if corpus_code == "sfu":
        subdirname = dirName.split("/")[-1]

        aux = special_codes[subdirname] if subdirname in special_codes.keys() else subdirname[0:2]
        prefix = f"{corpus_code}_{aux}"
    file_id = f"{prefix}_{fname}"
    return file_id


def gather_tool_sent_features(tool, sent_feat_single_path, genre, corpus_code):
    df_tool_gral = pd.DataFrame()
    for dirName, subdirList, fileList in os.walk(sent_feat_single_path):
        for fname in fileList:
            # codificar nombre file
            aux_fname = fname.replace(f".{tool}_counts", "").replace(".txt", "")
            file_id = codify_file_name(dirName, aux_fname, corpus_code)
            # leer cada archivo añadir columna id y tag
            df_sents_features = pd.read_csv(dirName + "/" + fname, sep=" ")
            conflict_column = 'Unnamed: 0'
            if conflict_column in df_sents_features.columns:
                df_sents_features.drop(columns=[conflict_column], inplace=True)

            # Para guardar columnas luego con orden id - cols -label
            cols_before = df_sents_features.columns.tolist()
            # Conseguir 'du2_WSJ920203-0038_sent0'
            df_sents_features["id_sent"] = df_sents_features.index.astype("str")
            df_sents_features["id_sent"] = file_id + "_sent" + df_sents_features["id_sent"]

            df_sents_features["label"] = genre

            new_cols = ["id_sent"] + cols_before + ["label"]
            df_sents_features = df_sents_features[new_cols]

            df_tool_gral = df_tool_gral.append(df_sents_features)

    return df_tool_gral


def general_sent_feats_gathering():
    for corpora_type, corpora in super_corpora.items():
        for tool in ["caevo", "free"]:
            df_corpora_type = pd.DataFrame()
            for genre, sub_corpora in corpora.items():
                for corpus in sub_corpora:
                    sent_feat_single_path = f"integer/corpora_outs/corpus_{corpora_type}/{genre}/{tool}_{corpus}/single_docs"
                    corpus_code = corpus[0:2] + corpus[-1] if "duc" in corpus else corpus[0:3]
                    tool_corpus_sent_features = gather_tool_sent_features(tool, sent_feat_single_path, genre,
                                                                          corpus_code)
                    df_corpora_type = df_corpora_type.append(tool_corpus_sent_features)

            # Guardar en archivo
            feat_file_tool_name = f"sent_features_{tool}_{corpora_type}_with_tags.csv"
            path = os.path.join(sent_features_gral_path, feat_file_tool_name)
            # GENERA
            df_corpora_type.to_csv(path, sep=" ", index=False)
