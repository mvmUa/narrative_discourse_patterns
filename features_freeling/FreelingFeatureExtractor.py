#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Count the vocabulary of a corpus of docs processed with Freeling.
'''
import joblib as joblib
import os
import numpy as np
import pandas as pd
import datetime
import traceback
import re


def save_error_exception(excep, log_file, dirName, fname):
    with open(log_file, 'a') as f:
        f.write("Hora: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
        f.write("path: " + os.path.join(dirName, fname) + "\n")
        f.write(str(excep) + "\n")
        f.write(traceback.format_exc() + "\n")
        f.write("***************************************************\n")


class FreelingFeatureExtractor:
    """
    FFE
    An extractor can act on a corpus or on a set of corpora.
    It generates
     - a directory with the features per line of each file
     - a file with the features of the set of documents, one doc in each line
    """
    DIR_SINGLE = "single_docs"

    def __init__(self, dir_free_path, root_salidas_FFE, decimal_separator="",
                 save_partials=False, dir_examples="", DMarkers_file=""):
        """
        :param dir_free_path: dir where are the folders with freeling files for a corpus, which can be a corpus (sfu) or several (lym,dormir)
        :param root_outputs_FFE: where the output directory of the Extractor feature will be stored, where results are stored
        :param decimal_separator: because openoffice separates with "," the decimals, in case we want to format the output with ",", indicate it
        :param save_partials:
        """
        self.dir_free_genre_path = dir_free_path
        self.decimal_separator = decimal_separator
        self.save_partials = save_partials
        self.root_salidas_FFE = root_salidas_FFE
        self.dir_examples = dir_examples

        # cargar lexicon discourse_markers
        with open(DMarkers_file, "r") as fr:
            self.DM_lexicon = [x for x in fr.read().split("\n") if x != ""]

    def create_FFE_dirs_for_subcorpora(self, subcorpora):
        """
        :param subcorpora: [lym,dormir] o [sfu]
        :return:
        """
        # Root dir for outputs
        self.dir_salidas_copora = {}
        for corpus in subcorpora:

            self.dir_salidas_copora[corpus] = os.path.join(self.root_salidas_FFE, f"free_{corpus}")
            self.create_dir(self.dir_salidas_copora[corpus])

            # Root dir for single stats results if necesary
            if self.save_partials:
                self.path_single_stats = os.path.join(self.dir_salidas_copora[corpus], self.DIR_SINGLE)
                self.create_dir(self.path_single_stats)

        # Root dir for examples
        if self.dir_examples != "":
            self.create_dir(self.dir_examples)

    def get_general_file_out(self):
        # Create a directory for several corpues
        general_file_out = os.path.join(self.dir_salidas_copora[self.nomCorpus],
                                        f"gral_caevo_stats_{self.nomCorpus}.csv")

        return general_file_out

    def get_participle_source(self, dataLine):
        """
        100 were be VBD VBD pos=verb|vform=past - 02604760-v )                              98 SUB - - - - - - - - - - - - - - - - - - - - - - -
        101 not not RB RB pos=adverb|type=general - 00024073-r (adv:101))))                100 ADV - - - - - - - - - - - - - - - - - - - - - - AM-NEG
        102 deceived deceive VBN VBN pos=verb|vform=participle - 02575082-v (vb-chunk:102) 100

        for the participle we look for its antecedent : 100
         # headers = ["numlinea","word","lema","morf1","morf_short","syn1","ner","synset","chunk","dep_number","dep_tag","coref","action"]
        """

        count = {"have": 0, "be": 0}
        # Get the lines in which participles appear.
        # We are interested in their index as well, the 0-based integer.
        participles_refs = dataLine[dataLine.syn1.str.contains("participle")]

        for index, row in participles_refs.iterrows():
            # when the participle is root, it does not refer to anyone, row-dep_number is 0 
            if int(row.dep_number) != 0:
                ref_verb_index = index - (row.numlinea - int(row.dep_number))

                aux = dataLine.loc[ref_verb_index]

                if aux.lema in count.keys():
                    count[aux.lema] += 1

        return count["have"], count["be"]

    def get_line_frequencies(self, dataLine):

        punct_tags = list(".,;:-_'?¿!")
        original_sentence = " ".join(dataLine.word.tolist())
        examples_dict = self.create_examples_dict()
        counts = {}

        # number of tokens that are not punctuation
        counts["words"] = len(dataLine[~dataLine.lema.isin(punct_tags)])

        # number of synset-adjetivos  02604760-a
        # proporcion de adjetivos respecto a los sustantivos
        # counts["synsetAdj"] = len(dataLine[dataLine.synset.str.endswith("-a")])
        counts["adjs"] = len(dataLine[dataLine.synset.str.endswith("-a")])

        # number of sustantivos
        counts["nouns"] = len(dataLine[dataLine.morf_short.str.startswith("N")])

        # number of adverbios synset-r
        s = dataLine[dataLine.synset.str.endswith("-r")]
        counts["advs"] = len(s)

        # determinante indefinido a an
        # pos = determiner
        s = dataLine[dataLine.syn1.str.contains("pos=determiner") & dataLine.lema.isin(["a"])]
        counts["indef_art"] = len(s)

        # number of pronombres 3a persona [he,she,they, his, hers, their, himself, herself]  He       he      PRP PRP pos=pronoun|type=personal
        s = dataLine[dataLine.lema.isin("he,she,they,his,hers,their,himself,herself".split(","))]
        counts["pers_pronouns_3P"] = len(s)

        # pronombres primera persona [i, we, me, us, our, ours]
        s = dataLine[dataLine.lema.isin("i,we,me,us,our,ours,myself,ourselves".split(","))]
        counts["pers_pronouns_1P"] = len(s)

        # pronombres 2a persona [you, yours]
        counts["pers_pronouns_2P"] = len(dataLine[dataLine.lema.isin("you,yours,yourself,yourselves".split(","))])

        # pronombres 2a persona [you, yours]
        counts["pers_pronouns_IT"] = len(dataLine[dataLine.lema.isin(["it"])])

        # pos=verb|vform=past
        # verbos en pasado
        counts["vform_past"] = len(dataLine[dataLine.syn1.str.endswith("past")])

        # pos=verb|vform=personal (presente, puede ser 1a o 2a persona
        # pos=verb|vform=personal|person=3 presente, 3a persona
        # incluimos las ocurrencias de may y can
        s = dataLine[dataLine.morf1.isin(["VBZ", "VBP"]) | (
                dataLine.morf1.isin(["MD"]) & dataLine.word.str.lower().isin(["can", "may"]))]
        counts["vform_present"] = len(s)

        # verb in 3erd person present
        # Por estar can y may etiquetados como MD, He can no aparecería
        counts["vform_present_3erd"] = len(dataLine[dataLine.morf1.isin(["VBZ"])])

        # "have + participle"
        # "be + participle
        counts["participle_have"], counts["participle_be"] = self.get_participle_source(dataLine)
        # examples gathering:
        for i in ["participle_have", "participle_be"]:
            if counts[i] > 0:
                examples_dict[i] = [original_sentence]

        # future / modal + will
        # pos=verb|type=modal
        s = dataLine[(dataLine.syn1.str.contains("type=modal")) & (dataLine.lema.str == "will")]
        counts["vform_future"] = len(s)

        for type_verb in "gerund,infinitive,participle".split(","):
            # gerund
            # pos=verb|vform=gerund    pos=verb|vform=infinitive    pos=verb|vform=participle
            s = dataLine[dataLine.syn1 == f"pos=verb|vform={type_verb}"]
            counts[f"vform_{type_verb}"] = len(s)

        # modal
        # type=modal  (any of can,may,must,ought,shall,should,would) excluding will
        # s = dataLine[dataLine.syn1 == "pos=verb|type=modal"]
        # cambio para excluir will
        s = dataLine[(dataLine.syn1.str.contains("type=modal")) & (dataLine.lema.str != "will")]
        counts["verb_modal_no_will"] = len(s)

        # Nombres propios
        # NP  pos=noun|type=proper|neclass=person
        nps = dataLine[dataLine.morf_short.str.contains("NP")]
        counts["noun_proper"] = len(nps)

        # NER
        # B-LOC , B-MISC, B-ORG, B-PER
        ner_exclude = ['The', 'And', 'What', 'She', 'Both', 'Another', 'Theres', 'If', 'When', 'Which', 'Her', 'These',
                       'Those',
                       'Then', 'They', 'This', 'Though', 'There', 'That', 'The', 'Would', 'Was', 'Where', 'While', 'We',
                       'Which', 'When',
                       'What', 'With', 'Who', 'Whose', 'Without', 'Thus', 'LRB', 'RRB']
        ner_in = dataLine[dataLine.ner.str.startswith("B-")]
        counts["ner"] = len([x for x in ner_in.word.tolist() if x not in ner_exclude])

        # numbers
        # Z     pos = number
        counts["numbers"] = len(dataLine[dataLine.morf_short.str.contains("Z")])

        # predicative complement (atributo) implica una estructura X is Y "the pan was gorgeous" "we were impressed"
        aux_pred = dataLine[dataLine.dep_tag.str.contains("PRD")]
        counts["predicative_complement"] = len(aux_pred)
        # example
        if len(aux_pred) != 0:
            examples_dict["predicative_complement"] = aux_pred.word.tolist()

        # wh-pronuuns What who
        aux_whpronoun = dataLine[dataLine.lema.isin(["what", "who"]) & dataLine.morf1.isin(["WP"])]
        counts["wh-pronoun"] = len(aux_whpronoun)
        # example
        if len(aux_pred) != 0:
            examples_dict["wh-pronoun"] = aux_whpronoun.word.tolist()

        # wh-adverbs
        aux_wh_adverb = dataLine[dataLine.morf1.isin(["WRB"])]
        counts["wh-adverb"] = len(aux_wh_adverb)
        # example
        if len(aux_pred) != 0:
            examples_dict["wh-adverb"] = aux_wh_adverb.word.tolist()

        # When - adverb
        feat = "when-adverb"
        feat_df_aux = dataLine[dataLine.morf1.isin(["WRB"]) & dataLine.lema.isin(["when"])]
        counts[feat] = len(feat_df_aux)
        # example
        if len(feat_df_aux) != 0:
            examples_dict[feat] = feat_df_aux.word.tolist()

        #  hyphens as - --	punctuation, digression (inc-mk) in freeling
        # https://github.com/TALP-UPC/FreeLing/blob/master/doc/grammars/enCHUNKtags
        hyphen_datalines = dataLine[dataLine.syn1.str.contains("type=hyphen") & dataLine.chunk.str.contains("inc-mk")]
        counts["hyphen_digression"] = len(hyphen_datalines)
        if counts["hyphen_digression"] > 0:
            examples_dict["hyphen_digression"] = [original_sentence]

        # Exclamative/interrogative sentences
        counts["sent_exclamative"] = len(dataLine[dataLine.syn1.str.contains("type=exclamationmark")])
        counts["sent_interrogative"] = len(dataLine[dataLine.syn1.str.contains("type=questionmark")])

        # parenthesis sentences  ( { [
        counts["sent_parenthesis"] = len(
            dataLine[dataLine.syn1.str.contains("punctenclose=close") & dataLine.lema.isin(["("])])

        # Example
        for i in ['sent_exclamative', 'sent_interrogative', 'sent_parenthesis']:
            if counts[i] > 0:
                examples_dict[i] = [original_sentence]

        # quotation marks
        quots = "``,'',\",“,”"
        counts["quotation_marks"] = len(
            dataLine[dataLine.syn1.str.contains("type=quotation") & dataLine.lema.isin(quots.split(","))])

        # LONG LEMAS+CONTENT WORDS (no cuenta ners): because we don't want verb conjugations to be counted as longer.
        # count word lengths: WARNING with the ner, remove "_".
        lemas_content_in_line = dataLine[~dataLine.synset.isin(["-"])].lema.tolist()

        for i in range(1, 7):
            words_i = [x for x in lemas_content_in_line if len(x) == i]
            counts["lema_length_" + str(i)] = len(words_i)
            if len(words_i) > 0:
                examples_dict["lema_length_" + str(i)] = words_i

        words_more_7 = [x for x in lemas_content_in_line if len(x) >= 7]
        counts["lema_length_min_7"] = len(words_more_7)
        if len(words_more_7) > 0:
            examples_dict["lema_length_min_7"] = words_more_7
        # -------------------------------------------------------
        # Extra frases introductorias, comas, NERS
        self.intr_phrases_checking(counts, dataLine, examples_dict)
        self.count_comas(counts, dataLine)
        self.count_NER_lenghts(counts, ner_in, examples_dict, ner_exclude)
        self.count_NER_type(counts, ner_in, examples_dict, ner_exclude)

        # DiscM
        words = " ".join([str(x).lower() for x in dataLine.word.tolist()])
        # Esto devuelve una lista con todas las coincidendias del pattern en a: ['There,', 'there ', 'there.']
        coincidences = [re.findall(rf"{x}[\.|,| |:|;]", words, flags=re.IGNORECASE) for x in self.DM_lexicon]
        # Cmo el resultado se supone que va a ser un array de arrays
        coincidences = np.concatenate(coincidences)
        counts['DiscM'] = len(coincidences)
        # EJEMPLOS
        if len(coincidences) > 0:
            examples_dict["DiscM"] = coincidences

        return counts, examples_dict

    def get_features_from_counts_per_document(self, count_df):
        """
        Features/ counts by a freeling file. calculates proportions and calculated features
        param count_df: dataset with the counts of the document. Each sentence of the document, one row
        :return: dictionary with the features for a document
        """
        features = {}
        total_num_sentences = len(count_df)
        # Existence a todos los elementos
        for col in count_df:
            features[f"E_{col}"] = len(count_df[count_df[col] > 0])
            features[f"E_{col}_prop"] = (features[f"E_{col}"] / total_num_sentences) * 100

        # Suma de elementos
        total_sum = count_df.sum()
        for colum in count_df.columns:
            features[colum] = total_sum[colum]

        #   Averaged de averaged
        count_df_average_considering_sents = count_df.apply(lambda x: x / x["words"] * 100, axis=1)
        count_df_average_averaged = count_df_average_considering_sents.mean()

        for colum in count_df_average_averaged.index:
            features[f"{colum}_avg"] = count_df_average_averaged[colum]

        # adj frente a sustantivos
        features["adj_nouns"] = total_sum.adjs / (total_sum.nouns if total_sum.nouns else 1)

        features["NP_nouns"] = total_sum.noun_proper / (total_sum.nouns if total_sum.nouns else 1)

        features["NER_NP"] = total_sum.ner / (total_sum.noun_proper if total_sum.noun_proper else 1)

        # cual es el tiempo verbal que predomina past,present,future
        features["predominant_time"] = total_sum[['vform_future', 'vform_past', 'vform_present']].idxmax()

        # cual es el pronombre personal predominante
        features["predominant_pers_pron"] = total_sum[
            ['pers_pronouns_1P', 'pers_pronouns_2P', 'pers_pronouns_3P']].idxmax()

        # cual es el tipo de NER predominante
        features["predominant_NER_type"] = total_sum[['B-LOC', 'B-MISC', 'B-ORG', 'B-PER']].idxmax()

        return features

    def count_Freeling_from_fre_file(self, file, corpus):
        #                2      is     be    VBZ        VBZ     pos=verb  -  02604760-v (vb-be:2)    0       ROOT         -     be.00
        headers = ["numlinea", "word", "lema", "morf1", "morf_short", "syn1", "ner", "synset", "chunk", "dep_number",
                   "dep_tag", "coref", "action"]
        data = pd.read_csv(file, delimiter='\s+', usecols=(range(0, 13)), header=None, names=headers)

        # obtener un dataframe por oracion
        # cada oracion comienza en numLinea1. Ontener los indices.
        sentence_ini = data[data.numlinea == 1].index.values.astype(int)
        df_sentence_list = []
        number_of_sentences = len(sentence_ini)
        for i in range(0, number_of_sentences):
            if i == number_of_sentences - 1:
                df_sentence_list.append(data.iloc[sentence_ini[i]:])
            else:
                df_sentence_list.append(data.iloc[sentence_ini[i]:sentence_ini[i + 1]])

        # En un archivo se guardan las frecuencias sin más, una linea por oracion
        counts = []
        num_sents_analized = 0
        for dataLine in df_sentence_list:
            # Obtener counts de una oracion desde archivo freeling
            counts_line, examples_dict = self.get_line_frequencies(dataLine)
            counts.append(counts_line)
            # Guardar los ejmplos si procede
            if self.dir_examples != "":
                self.save_examples(examples_dict, dataLine, file, num_sents_analized, corpus)
            num_sents_analized += 1

        # Tengo todos los counts en un df
        count_df = pd.DataFrame(counts)
        # las features que se pueden salvar para una sola linea
        line_features = count_df.columns

        if self.save_partials:
            ext = "free_counts"
            file_name = file.split("/")[-1].replace("fre_out", ext)
            if "sfu" in file:
                dir_path = os.path.join(self.dir_salidas_copora[corpus], self.DIR_SINGLE, file.split("/")[-2])
                self.create_dir(dir_path)

                file_name = file.split("/")[-2] + "/" + file_name
            file_stats_out = os.path.join(self.dir_salidas_copora[corpus], self.DIR_SINGLE, file_name)
            count_df.to_csv(file_stats_out, sep=' ')

        features_dict = {}
        # Calcular las features. Features es un diccionario
        # **************************************** COMENTADO PARA CALCULAR EJEMPLOS /SAVEPARTS
        features_dict = self.get_features_from_counts_per_document(count_df)

        return features_dict

    def freeling_cheking_to_corpus(self, set_type, genre, corpus):
        """
        Calculate the features for all the docus in a corpus.

        :param set_type: [train o test]
        :param genre: [news, tales, reviews ]
        :param corpus: lym
        :return: dataframe with features from one corpus
        """
        # files_to_review = ['AP900615-0164.fre_out']
        log_file = os.path.join(self.dir_salidas_copora[corpus],
                                f'log_{datetime.datetime.now().strftime("%d%m_%H%M%S")}.txt')
        count = 0
        features = []

        fre_corpus_path = os.path.join(self.dir_free_genre_path, f"free_{corpus}")

        already_accounted_files = []
        path_fre_corpus_features_jolib = os.path.join(self.dir_salidas_copora[corpus], f"{corpus}_free_features.joblib")
        if os.path.exists(path_fre_corpus_features_jolib):
            already_accounted_files = joblib.load(path_fre_corpus_features_jolib)[["file"]].file.tolist()

        for dirName, subdirList, fileList in os.walk(fre_corpus_path):
            lista_files = [fl for fl in fileList if fl.replace(".fre_out", "") not in already_accounted_files]
            for fname in fileList:

                # fname = "bo_msd_b_51.fre_out"
                doc_name = fname
                # ********************************************************
                # PARA EJEMPLOS, %= DOCS DE CADA CORPUS

                try:

                    count += 1
                    file_name = os.path.join(dirName, fname)
                    print(
                        f"{count}: {file_name.replace('corpora_analyzed/', '')}")

                    # Guardamos como columna del df el nombre de archivo
                    # Con sfu un paso más, porque el nombre solo no sirve, necesitamos tb el dir
                    if "sfu" in dirName:
                        # modificamos para legibilidad en analisis de features cook/yes13 --> cook_yes13
                        fname = dirName.split("/")[-1] + "_" + fname
                    # incluir el nombre de archivo
                    complete_doc_features = {"file": fname.replace(".fre_out", "")}

                    # Calcular las features para un solo archivo
                    # doc_features es un diccionario
                    doc_features = self.count_Freeling_from_fre_file(file_name, corpus)

                except RuntimeError as e:
                    print("Error")
                    print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                    save_error_exception(e, log_file, dirName, fname)

                except UnicodeDecodeError as e:
                    print("unicode Error")
                    print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                    save_error_exception(e, log_file, dirName, fname)

                except Exception as e:
                    print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                    save_error_exception(e, log_file, dirName, fname)

        features_df = pd.DataFrame(features)

        return features_df

    def corpora_features_gathering(self, set_type, genre, short_corpora):
        """
        Calculates the features of the corpora corpora corpus from freeling files
        generates an excel file with the data and statistics
        :param set_type: [train o test]
        :param genre: [news, tales, reviews ]
        :param short_corpora: [lym, ...]
        :return:
        """
        column_order = []
        corpora_features = {}
        # short_corpora ["lym","andersen"]
        for corpus in short_corpora:
            print_time_message(f"Comienza el corpus {corpus}")
            features_df = self.freeling_cheking_to_corpus(set_type, genre, corpus)

            path_fre_corpus_features = os.path.join(self.dir_salidas_copora[corpus], f"{corpus}_free_features.csv")
            # GENERA comentado
            features_df.to_csv(index=False, path_or_buf=path_fre_corpus_features, columns=column_order, sep=":")
            corpora_features[corpus] = features_df

    def freeling_functions_to_corpora(self, set_type, genre, short_corpora, funciones):
        column_order = ['file', 'words', 'sentences',
                        'sent_exclamative', 'sent_exclamative_prop', 'sent_interrogative', 'sent_interrogative_prop',
                        'sent_parenthesis',
                        'predominant_pers_pron', 'pers_pronouns_1P', 'pers_pronouns_2P', 'pers_pronouns_3P',
                        'pers_pronouns_IT', 'It_pronoun_per_sent', 'E_1P', 'E_1P_prop', 'E_2P', 'E_2P_prop',
                        'ner', 'E_NER', 'E_NER_prop', 'NER_NP', 'nouns', 'noun_proper', 'noun_proper_uniq', 'NP_nouns',
                        'E_NP', 'E_NP_prop',
                        'numbers', 'E_Numbers', 'E_Numbers_prop', 'numbers_words',
                        'synsetAdj', 'E_adj', 'E_adj_prop', 'adj_sent', 'adj_sust', 'adj_words',
                        'vform_future', 'vform_past', 'vform_present', 'vform_present_3erd', 'predominant_time',
                        'vform_gerund', 'vform_infinitive', 'vform_participle', 'verb_modal',
                        'participle_be', 'participle_have', 'predicative_complement',
                        'quotation_marks', 'E_quotation_marks',
                        'synsetAdv', 'wh-adverb', 'wh-pronoun', 'when-adverb',
                        'lema_length_1', 'lema_length_2', 'lema_length_3', 'lema_length_4', 'lema_length_5',
                        'lema_length_6', 'lema_length_min_7']

        corpora_counts = {}
        for corpus in short_corpora:
            corpora_counts[corpus] = self.freeling_functions_to_corpus(set_type, genre, corpus, funciones)
            # guardar las features de un corpus
            #   corpus + feat_type + corpus_features_sufix
            path_fre_corpus_features = os.path.join(self.dir_salidas_copora[corpus],
                                                    f"{corpus}_free_features_INTR_PHRA.csv")
            # GENERA
            corpora_counts[corpus].to_csv(index=False, path_or_buf=path_fre_corpus_features, sep=":")

    def freeling_functions_to_corpus(self, set_type, genre, corpus, funciones):

        log_file = os.path.join(self.dir_salidas_copora[corpus],
                                f'log_{datetime.datetime.now().strftime("%d%m%H%M%S")}.txt')
        count = 0
        counts = []

        fre_corpus_path = os.path.join(self.dir_free_genre_path, f"free_{corpus}")

        for dirName, subdirList, fileList in os.walk(fre_corpus_path):
            # dictionary/dataframe per corpus

            for fname in fileList:
                doc_name = fname
                if corpus in ["sfu", "duc"] or isInSelectedTales(fname):
                    try:

                        count += 1
                        file_name = os.path.join(dirName, fname)
                        print(file_name)
                        # occurrences es un diccionario
                        occurrences = self.freeling_functions_to_fre_file(file_name, corpus, funciones)

                        if "sfu" in dirName:
                            fname = dirName.split("/")[-1] + "_" + fname
                        # incluir el nombre de archivo
                        complete_doc_features = {"file": fname.replace(".fre_out", "")}

                        complete_doc_features.update(occurrences)
                        counts.append(complete_doc_features)

                    except RuntimeError as e:
                        print("Error")
                        print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                        save_error_exception(e, log_file, dirName, fname)

                    except UnicodeDecodeError as e:
                        print("unicode Error")
                        print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                        save_error_exception(e, log_file, dirName, fname)

                    except Exception as e:
                        print("*****  id_assignment: " + str(os.path.join(dirName, fname)) + "\n")
                        save_error_exception(e, log_file, dirName, fname)
        return pd.DataFrame(counts)

    def freeling_functions_to_fre_file(self, file, corpus, funciones):
        #                2      is     be    VBZ        VBZ     pos=verb  -  02604760-v (vb-be:2)    0       ROOT         -     be.00
        headers = ["numlinea", "word", "lema", "morf1", "morf_short", "syn1", "ner", "synset", "chunk", "dep_number",
                   "dep_tag", "coref", "action"]
        data = pd.read_csv(file, delimiter='\s+', usecols=(range(0, 13)), header=None, names=headers)

        metodos = []
        for funcion in funciones:
            metodos.append(getattr(self, funcion))

        # obtener un dataframe por oracion
        # cada oracion comienza en numLinea1. Ontener los indices.
        sentence_ini = data[data.numlinea == 1].index.values.astype(int)
        df_sentence_list = []
        number_of_sentences = len(sentence_ini)
        for i in range(0, number_of_sentences):
            if i == number_of_sentences - 1:
                df_sentence_list.append(data.iloc[sentence_ini[i]:])
            else:
                df_sentence_list.append(data.iloc[sentence_ini[i]:sentence_ini[i + 1]])

        # En un archivo se guardan las frecuencias sin más, una linea por oracion
        counts = {}
        counts["numSent"] = number_of_sentences
        counts["numSent"] = number_of_sentences
        for dataLine in df_sentence_list:
            for metodo in metodos:
                metodo(counts, dataLine)

        return counts

    def intr_phrases_checking(self, counts, df_fre_lines, example_dict):
        lemas = df_fre_lines.lema.tolist()[1:4]
        if "intr_ph" not in counts.keys():
            counts["intr_ph"] = 0
            counts["intr_with_PN_NER"] = 0
            counts["intr_with_adverb"] = 0

        if "," in lemas:
            # +2 para qe incluya la coma en la salida
            pos_comma = lemas.index(",") + 2
            intr_lines = df_fre_lines[0:pos_comma]
            intr_phrase = " ".join(intr_lines.word.tolist())
            counts["intr_ph"] += 1
            c_aux = len(intr_lines[intr_lines.morf_short.str.contains("NP")]) + len(
                intr_lines[intr_lines.ner.str.startswith("B-")])
            counts["intr_with_PN_NER"] += 1 if c_aux > 0 else 0
            counts["intr_with_adverb"] += 1 if len(intr_lines[intr_lines.synset.str.endswith("-r")]) > 0 else 0

            for i in ['intr_ph', 'intr_with_PN_NER', 'intr_with_adverb']:
                if counts[i] > 0:
                    example_dict[i] = [intr_phrase]

    def count_comas(self, counts, df_fre_lines):
        if "c_commas" not in counts.keys():
            counts["c_commas"] = 0
        comas = df_fre_lines[df_fre_lines.lema == ","]

        counts["c_commas"] += len(comas)

    def count_NER_lenghts(self, counts, ner_lines, examples_dict, ner_exclude):
        for i in range(1, 5):
            counts["NER_length_" + str(i)] = 0
        counts["NER_length_5_or_more"] = 0

        for id, row in ner_lines.iterrows():
            if str(row.word) not in ner_exclude:
                length = len(str(row.lema).split("_"))
                if length < 5:
                    counts["NER_length_" + str(length)] += 1
                    if length > 1:
                        examples_dict["NER_length_" + str(length)].append(row.word)

                else:
                    counts["NER_length_5_or_more"] += 1
                    examples_dict["NER_length_5_or_more"].append(row.word)

    def count_NER_type(self, counts, ner_lines, example_dict, ner_exclude):
        for type_ner in "B-LOC,B-PER,B-ORG,B-MISC".split(","):
            ners = ner_lines[ner_lines.ner.isin([type_ner])]
            counts[type_ner] = len(ners)
            if counts[type_ner] > 0:
                example_dict[type_ner] = ners.word.tolist()

    def create_dir(self, pathOut):
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

    def save_examples(self, examples_dict, dataLine, file, num_sents_analized, corpus):

        interesting_features_for_examples = ['participle_have', 'participle_be', 'predicative_complement', 'wh-pronoun',
                                             'wh-adverb', 'when-adverb',
                                             'sent_exclamative', 'sent_interrogative', 'sent_parenthesis',
                                             'hyphen_digression',
                                             'intr_ph', 'intr_with_PN_NER', 'intr_with_adverb', 'indef_art',
                                             'B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'DiscM']
        interesting_length_features_for_examples = ['lema_length_1', 'lema_length_2', 'lema_length_3',
                                                    'lema_length_4', 'lema_length_5', 'lema_length_6',
                                                    'lema_length_min_7',
                                                    'NER_length_2', 'NER_length_3', 'NER_length_4',
                                                    'NER_length_5_or_more']

        original_sentence = " ".join([str(x) for x in dataLine.word.tolist()])

        # incorporar ejemplos de features existentes a archivos correspondientes
        for feat in [x for x in examples_dict.keys() if len(examples_dict[x]) != 0]:
            df_aux = pd.DataFrame.from_dict({"corpus": [corpus],
                                             "file": [file],
                                             "sent_number": [num_sents_analized],
                                             "original_sentence": [original_sentence],
                                             "element": ",".join(examples_dict[feat])}, orient="columns")

            features_file_name = os.path.join(self.dir_examples, f"{feat}.joblib")
            if os.path.exists(features_file_name):
                df = joblib.load(features_file_name)
                df = df.append(df_aux)
            else:
                df = df_aux

            joblib.dump(value=df, filename=features_file_name, compress="gzip")
            df.to_csv(features_file_name.replace("joblib", "csv"))

    def create_examples_dict(self):
        interesting_features_for_examples = ['participle_have', 'participle_be', 'predicative_complement', 'wh-pronoun',
                                             'wh-adverb', 'when-adverb',
                                             'sent_exclamative', 'sent_interrogative', 'sent_parenthesis',
                                             'hyphen_digression',
                                             'intr_ph', 'intr_with_PN_NER', 'intr_with_adverb', 'indef_art',
                                             'B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'DiscM']
        interesting_length_features_for_examples = ['lema_length_1', 'lema_length_2', 'lema_length_3',
                                                    'lema_length_4', 'lema_length_5', 'lema_length_6',
                                                    'lema_length_min_7',
                                                    'NER_length_2', 'NER_length_3', 'NER_length_4',
                                                    'NER_length_5_or_more']
        all = interesting_features_for_examples + interesting_length_features_for_examples

        return {x: [] for x in all}


#  Hasta aquí la clase FreelingFeatureExtractor
############################################################################################


####################################     UTILS                 ########################################################

def dfs_dict_to_excel(file_name, dfs_dic):
    with pd.ExcelWriter(file_name + '.xlsx') as writer:
        # quiero una hoja de excel por elemeto del diccionario
        for name, df in dfs_dic.items():
            index = False
            if "_py_stats" in name:
                index = True  # para guardar mean, max, ...
            df.to_excel(writer, sheet_name=name, index=index)


def ordenar_columnas_and_to_excel_ad_hoc():
    '''
    Adhoc para reordenar columnas en excel
    '''
    column_order = ['file', 'words', 'sentences',
                    'sent_exclamative', 'sent_exclamative_prop', 'sent_interrogative', 'sent_interrogative_prop',
                    'sent_parenthesis',
                    'predominant_pers_pron', 'pers_pronouns_1P', 'pers_pronouns_2P', 'pers_pronouns_3P',
                    'pers_pronouns_IT', 'It_pronoun_per_sent', 'E_1P', 'E_1P_prop', 'E_2P', 'E_2P_prop',
                    'ner', 'E_NER', 'E_NER_prop', 'NER_NP', 'nouns', 'noun_proper', 'noun_proper_uniq', 'NP_nouns',
                    'E_NP', 'E_NP_prop',
                    'numbers', 'E_Numbers', 'E_Numbers_prop', 'numbers_words',
                    'synsetAdj', 'E_adj', 'E_adj_prop', 'adj_sent', 'adj_sust', 'adj_words',
                    'vform_future', 'vform_past', 'vform_present', 'vform_present_3erd', 'predominant_time',
                    'participle_be', 'participle_have', 'predicative_complement',
                    'quotation_marks', 'E_quotation_marks',
                    'synsetAdv', 'wh-adverb', 'wh-pronoun', 'when-adverb',
                    'lema_length_0', 'lema_length_1', 'lema_length_2', 'lema_length_3', 'lema_length_4',
                    'lema_length_5', 'lema_length_6', 'lema_length_min_7']

    # Cargo todos los csv
    corpora_features = {}
    corpora_features["sfu"] = pd.read_csv("datasheets_free/features_df_sfu.csv")[column_order]
    corpora_features["duc"] = pd.read_csv("datasheets_free/features_df_duc.csv")[column_order]

    corpora_features["dormir"] = pd.read_csv("datasheets_free/features_df_dormir.csv")[column_order]
    corpora_features["lym"] = pd.read_csv("datasheets_free/features_df_lym.csv")[column_order]

    corpora_features["tales"] = pd.read_csv("datasheets_free/features_df_tales.csv")[column_order]

    for corpus_key in ["dormir", "lym", "sfu", "duc", "tales"]:
        corpora_features[corpus_key + "_py_stats"] = corpora_features[corpus_key].describe(include='all')
    # guardar en la misma excel
    file_name = "datasheets_free/freeling_features_corpora_with_stats_ordered"
    dfs_dict_to_excel(file_name, corpora_features)


def cargar_corpora_features_to_dfs_dict(dict_corpus_pathFeatures):
    # Cargo todos los csv
    corpora_features = {}

    for corpus, features_path in dict_corpus_pathFeatures.items():
        corpora_features[corpus] = pd.read_csv(features_path)
    return corpora_features


def conteo_general_integer(saverPart, dir_examples, DM_file, project_root):
    """
    module that executes the analysis for each part of the corpus
    :param saverPart: if we want to save partial results per sentence
    :param dir_examples: if we want to save examples of the features we are interested in, here the directory dd is saved
    return:
    """
    ###################################################
    ### Conteo de los diferentes elementos
    corpora_genres_test = {
        "news": ["duc2002"],
        "tales": ["lym", "andersen"],
        # "reviews": ["opin"]
        "reviews": ["msd", "opin"]
    }

    corpora_genres_train = {
        "news": ["duc2004"],
        # "news" : []
        "tales": ["lym", "dormir"],
        "reviews": ["sfu"]
    }
    super_corpora = {
        "test": corpora_genres_test,
        "train": corpora_genres_train
    }
    tool = "free"

    for corpora_type, corpora in super_corpora.items():
        for genre, sub_corpora in corpora.items():
            print_time_message(f"Comienza el GENERO {genre}")
            # ruta del directorio dd los arcivos de freeling, aunque cada subcorpus tenga su propia carpeta
            dir_free_path = f"{project_root}/integer/corpora_analyzed/corpus_{corpora_type}/{genre}"

            # ruta del directorio dd se tiene que crear el directorio de salidas del FFE
            dir_salidas_FvFE = f"{project_root}/integer/corpora_outs_normalized/corpus_{corpora_type}/{genre}"

            print(dir_salidas_FvFE)
            # Creamos una instancia del CvFE
            sd = FreelingFeatureExtractor(dir_free_path, dir_salidas_FvFE,
                                          decimal_separator="",
                                          save_partials=saverPart,
                                          dir_examples="",
                                          DMarkers_file=DM_file)
            # crear directorios salida
            sd.create_FFE_dirs_for_subcorpora(sub_corpora)
            # calcular las features
            sd.corpora_features_gathering(corpora_type, genre, sub_corpora)


def print_time_message(message):
    aux = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n")
    aux += message
    aux += "#" * 10
    print("\n" + aux + "\n\n")
