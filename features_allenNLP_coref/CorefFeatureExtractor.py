import pickle
import os
import sys
import numpy as np
import pandas as pd

dcoref_sample = {
    'top_spans': [[0, 1], [3, 3], [5, 8], [5, 12], [5, 14], [8, 12], [8, 14], [11, 12], [14, 14], [17, 17], [18, 22],
                  [21, 22], [24, 26], [27, 27], [33, 34], [36, 38], [41, 41], [46, 46], [48, 48], [50, 50], [51, 51],
                  [54, 56], [57, 57], [58, 58], [59, 59], [64, 64], [67, 67], [67, 71], [69, 71], [72, 72], [74, 74],
                  [75, 76], [78, 80], [85, 86], [85, 89], [88, 89], [92, 92], [94, 94], [96, 97], [100, 101],
                  [102, 102], [103, 103], [105, 105], [108, 108], [110, 112], [114, 114]],
    'predicted_antecedents': [-1, -1, -1, -1, -1, -1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 20, -1,
                              -1, 3, -1, -1, -1, -1, -1, -1, 16, 19, 21, -1, 20, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    'document': ['Two', 'bats', 'were', 'training', 'for', 'a', 'great', 'flying', 'competition', 'in', 'which', 'all',
                 'bats', 'took', 'part', '.', '\n', 'On', 'the', 'day', 'of', 'the', 'race', ',', 'the', 'smaller',
                 'bat', 'flew', 'incredibly', 'well', ',', 'clearly', 'deserving', 'the', 'prize', '.', 'The', 'other',
                 'bat', ',', 'although', 'very', 'disappointed', 'at', 'not', 'having', 'won', ',', 'ran', 'to',
                 'congratulate', 'him', ',', 'while', 'the', 'other', 'bats', 'started', 'criticising', 'him', 'or',
                 'went', 'away', 'in', 'anger', '.', '\n', 'Grateful', ',', 'the', 'new', 'champion', 'decided', 'to',
                 'share', 'the', 'prize', '.', 'The', 'little', 'bat', 'had', 'not', 'only', 'won', 'the', 'race',
                 'and', 'the', 'prize', ',', 'but', 'he', 'had', 'also', 'won', 'a', 'friend', '.', 'And', 'all',
                 'this', 'came', 'about', 'from', 'knowing', 'how', 'to', 'lose', 'in', 'a', 'sporting', 'manner', '.',
                 '\n'],
    'clusters': [[[0, 1], [11, 12], [54, 56]], [[36, 38], [51, 51], [59, 59]], [[33, 34], [75, 76], [88, 89]],
                 [[24, 26], [78, 80]], [[21, 22], [85, 86]], [[69, 71], [92, 92]]]
}


class CorefFeatureExtractor:
    """
    An extractor can act on a corpus or on a set of corpora.
    It generates
     - a directory with the features per line of each file
     - a file with the features of the set of documents, one doc in each line
    """
    DIR_SINGLE = "single_docs"

    def __init__(self, dir_coref_path, root_salidas_CoFE, decimal_separator="",
                 save_partials=False):
        """
        :param dir_free_path: dir where are the folders with allenNLP files for a corpus, which can be one corpus (sfu) or several (lym,sleep)
        :param root_outputs_CoFE: where the output directory of the Extractor feature, where results will be stored
        :param decimal_separator: because openoffice separates with "," the decimals, in case we want to format the output with ",", indicate it
        :param save_partials:
        """
        self.dir_coref_genre_path = dir_coref_path
        self.decimal_separator = decimal_separator
        self.save_partials = save_partials
        self.root_salidas_CoFE = root_salidas_CoFE

    def create_CoFE_dirs_for_subcorpora(self, subcorpora):
        # Root dir for outputs
        self.dir_salidas_copora = {}
        for corpus in subcorpora:
            self.dir_salidas_copora[corpus] = os.path.join(self.root_salidas_CoFE, f"coref_{corpus}")
            self.create_dir(self.dir_salidas_copora[corpus])

    def create_dir(self, pathOut):
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

    def compute_coref_features(self, clusters):

        dict_features = {}

        dict_features["chain_amount"] = len(clusters)

        chains_len = [len(a) for a in clusters]
        dict_features["mean_chain_len"] = 0 if dict_features["chain_amount"] == 0 else int(np.array(chains_len).mean())

        chain_spread = lambda x: x[len(x) - 1][0] - x[0][0]
        chains_spread = [chain_spread(x) for x in clusters]
        dict_features["chain_spread_mean"] = 0 if dict_features["chain_amount"] == 0 else int(
            np.array(chains_spread).mean())

        # chains with len > 2
        maximal_len_chains = [x for x in clusters if len(x) > 2]
        dict_features["maximal_len_chains_amount"] = len(maximal_len_chains)

        entity_concentration = [extension / ocurrencias for (extension, ocurrencias) in zip(chains_spread, chains_len)]
        dict_features["entity_concentration_mean"] = 0 if dict_features["chain_amount"] == 0 else int(
            np.array(entity_concentration).mean())

        return dict_features

    def features_for_corpus(self, corpus):
        features = []
        count = 0

        coref_corpus_path = os.path.join(self.dir_coref_genre_path, f"coref_{corpus}")

        for dirName, subdirList, fileList in os.walk(coref_corpus_path):

            for fname in fileList:
                try:

                    count += 1
                    file_name = os.path.join(dirName, fname)
                    dcoref = read_dict_from_pkl(file_name)
                    doc_features = self.compute_coref_features(dcoref["clusters"])
                    if corpus in ["sfu"]:
                        fn = file_name.split("/")[-2] + "_" + fname.replace(".pkl", "")
                    else:
                        fn = fname.replace(".pkl", "")
                    complete_doc_features = {"file": fn}
                    complete_doc_features.update(doc_features)
                    features.append(complete_doc_features)

                except RuntimeError as e:
                    print("Error")

                except UnicodeDecodeError as e:
                    print("unicode Error")
                except:
                    print(file_name)
                    print("Unexpected error:", sys.exc_info()[0])

        column_order = ["file", "chain_amount", "mean_chain_len", "chain_spread_mean", "maximal_len_chains_amount",
                        "entity_concentration_mean"]
        features_df = pd.DataFrame(features)
        features_df = features_df[column_order]
        return features_df

    def features_for_corpora(self, corpora):

        corpora_features = {}
        for corpus in corpora:
            features_df = self.features_for_corpus(corpus)

            path_coref_corpus_features = os.path.join(self.dir_salidas_copora[corpus], f"{corpus}_coref_features.csv")
            features_df.to_csv(index=False, path_or_buf=path_coref_corpus_features, sep=":")
            corpora_features[corpus] = features_df

        return corpora_features

    def isSuitable(self, corpus, fname):
        file_with_cicling_tales = "dormir_y_LyM.txt"

        with open(file_with_cicling_tales, "r") as t:
            tales = [tale.replace(".txt\n", "") for tale in t.readlines()]
        isInSelectedTales = lambda x: x in tales

        return (corpus in ["sfu", "duc"] or isInSelectedTales(fname.replace(".pkl", "").replace(".txt", "")))

    def cargar_CorporaCiclingFeatrures_DF(self, tipo):
        corpora_features = {}
        path = "/datasheets_coref/"
        corpora_features["sfu"] = pd.read_csv(path + "coref_features_df_sfu.csv")
        corpora_features["duc"] = pd.read_csv(path + "coref_features_df_duc.csv")

        corpora_features["dormir"] = pd.read_csv(path + "coref_features_df_dormir.csv")
        corpora_features["lym"] = pd.read_csv(path + "coref_features_df_lym.csv")

        corpora_features["tales"] = pd.read_csv(path + "coref_features_df_tales.csv")

        return corpora_features


def read_dict_from_pkl(my_file):
    with open(my_file, "rb") as mf:
        mydict = pickle.load(mf)
    return mydict


def dfs_dict_to_excel(file_name, dfs_dic):
    with pd.ExcelWriter(file_name + '.xlsx') as writer:
        for name, df in dfs_dic.items():
            index = False
            if "_py_stats" in name:
                index = True
            df.to_excel(writer, sheet_name=name, index=index)


def calcularMedida(medida, corpora_features):
    medidas = {
        "media": pd.DataFrame.mean,
        "moda": pd.DataFrame.mode,
        "max": pd.DataFrame.max,
        "min": pd.DataFrame.min,
        "suma": pd.DataFrame.sum
    }

    resultados = []
    corpora = ["tales", "sfu", "duc"]
    for corpus in corpora:
        resultados.append(corpora_features[corpus].loc[:, "chain_amount":].apply(medidas[medida], axis=0))

    resultados_df = pd.DataFrame(resultados)
    resultados_df.insert(0, "corpus", corpora)
    resultados_df = resultados_df.round(2)

    return resultados_df


def conteo_general_integer():
    corpora_genres_test = {
        "reviews": ["opin"]
    }

    corpora_genres_train = {
        "reviews": ["sfu"]
    }
    super_corpora = {
        "test": corpora_genres_test,
        "train": corpora_genres_train
    }
    tool = "coref"

    for corpora_type, corpora in super_corpora.items():
        for genre, sub_corpora in corpora.items():
            dir_coref_genre_path = f"corpora_analyzed/corpus_{corpora_type}/{genre}"

            dir_salidas_CoFE = f"corpora_outs/corpus_{corpora_type}/{genre}"

            print(dir_salidas_CoFE)
            sd = CorefFeatureExtractor(dir_coref_genre_path, dir_salidas_CoFE, decimal_separator="",
                                       save_partials=True)
            sd.create_CoFE_dirs_for_subcorpora(sub_corpora)
            sd.features_for_corpora(sub_corpora)
