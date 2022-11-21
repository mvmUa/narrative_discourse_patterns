#! /usr/bin/python3
import xml.etree.ElementTree as ET
import os
from enum import Enum
import numpy as np
import re
import nltk
import pandas as pd
import matplotlib
from matplotlib import pylab
from six import itervalues, text_type, add_metaclass


# CvFE
class CaevoFeatureExtractor:
    ns = "{http://chambers.com/corpusinfo}"

    class Tags(Enum):

        SENTENCE = 'sentence'
        TIMEX = 'timex'
        TOKEN = 't'
        EVENT = 'event'
        EVENTS = 'events'
        TLINK = 'tlink'
        ENTRY = 'entry'
        DEPS = 'deps'
        PARSE = 'parse'
        FILE = "file"

    counts_keys = ["sentences_iter", "events", "sentences_whole"]

    events_types = ["ASPECTUAL", "I_ACTION", "I_STATE", "OCCURRENCE", "PERCEPTION", "REPORTING", "STATE"]
    tlink_relation_types = ['BEFORE', 'AFTER', 'IBEFORE', 'IAFTER', 'INCLUDES', 'IS_INCLUDED', 'BEGINS', 'BEGUN_BY',
                            'ENDS', 'ENDED_BY', 'SIMULTANEOUS', 'NONE', 'VAGUE', 'UNKNOWN', 'OVERLAP',
                            'BEFORE_OR_OVERLAP', 'OVERLAP_OR_AFTER']

    timex_types = ["DATE", "TIME", "DURATION", "SET"]

    # son text
    subject_types = ["nsubj", "csubj", "xsubj"]
    object_types = ["dobj"]
    phrase_types = ["NP", "VP", "NNP"]

    counts = {}

    DIR_SINGLE = "single_docs"

    def __init__(self, dir_xml_path, dir_salidas_CvFE=None, nomCorpus=None, decimal_separator=None, save_partials=None):
        """
        param dir_xml_path: directory containing the documents OF A SINGLE CORPUS. May have hierarchies
        Count of the different categories that caevo_features provides and persistence in a text file.
        When each caevo_features document is in a separate file
        :param decimal_separator: because openoffice separates with "," the decimals, in case we want output formatting with ",", indicate it
        :param save_partials:
        :param root_outputs_CvFE: where the output directory of the Extractor feature will be saved, where results are saved.
        """
        self.dir_xml_path = dir_xml_path
        self.decimal_separator = decimal_separator
        self.save_partials = save_partials
        self.nomCorpus = nomCorpus
        self.root_salidas_CvFE = dir_salidas_CvFE

    def create_CvFE_dirs(self):
        """
        Creation of main out directories for the Extractor
        :return:
        """
        DIR_OUT = "caevoFeatureExtractor_out"
        DIR_GRAL_STATS = "general_stats"
        DIR_SINGLE = "single_docs"

        # Root dir for outputs
        self.create_dir(os.path.join(self.root_salidas_CvFE, self.DIR_OUT))

        # Root dir for general_stats
        self.path_gral_stats = os.path.join(self.root_salidas_CvFE, self.DIR_GRAL_STATS)
        self.create_dir(self.path_gral_stats)

        # Root dir for single stats results idf necesary
        self.path_logs = os.path.join(self.root_salidas_CvFE, "logs")
        self.create_dir(self.path_logs)

        # Root dir for single stats results idf necesary
        if self.save_partials:
            self.path_single_stats = os.path.join(self.root_salidas_CvFE, self.DIR_SINGLE)
            self.create_dir(self.path_single_stats)

    def conteo_corpus(self, nomCorpus):
        """
         Count of the different categories provided by caevo_features and persistence in a text file
        """
        numFiles = 0
        legend_saved = False

        general_file_out = self.get_general_file_out()

        for dirName, subdirList, fileList in os.walk(self.dir_xml_path):
            for fname in fileList:
                if fname.endswith(".xml"):
                    numFiles += 1

                    file_path = os.path.join(dirName, fname)
                    file_sub_path_aux = dirName.replace(self.dir_xml_path, "")
                    file_sub_path = "" if file_sub_path_aux == "" else file_sub_path_aux.split("/")[1]

                    root = self.parseXMLDoc(file_path)

                    files = [elem for elem in root.iter(self.ns + self.Tags.FILE.value)]

                    for file_node in files:

                        doc_name = file_node.get("name")

                        all_file_info, legend = self.conteo_docu(file_sub_path, file_node, doc_name=doc_name)

                        field_separator = ":"
                        with open(general_file_out, 'a') as the_file:
                            if not legend_saved:
                                the_file.write("file" + field_separator + field_separator.join(legend) + "\n")
                                legend_saved = True
                            if self.decimal_separator is not "":
                                all_file_info = [str(x).replace(".", self.decimal_separator) for x in all_file_info]
                            else:
                                all_file_info = [str(x) for x in all_file_info]

                            if nomCorpus in ["sfu"]:
                                doc_name = file_sub_path + "_" + doc_name

                            all_file_info = doc_name + field_separator + field_separator.join(all_file_info)
                            the_file.write(all_file_info + "\n")

    def conteo_docu(self, file_sub_path, root, doc_name=""):
        """
        Counting elements of the whole document
        :return: List with the accounts for this document and list with the name of the corresponding fields
        """

        # Each entry keeps information regarding one sentence
        entries = [elem for elem in root.iter(self.ns + self.Tags.ENTRY.value)]

        doc_counts = {}
        doc_counts["c_sentences"] = len(entries)

        counts_to_save = []

        for entry in entries:
            # for each sentence
            sentence_counts = self.conteo_sentence(entry)

            # recover de list of indicators (0/1) per sentence that will allow aus to determine % of sentences with X element
            self.set_sentence_indicators(sentence_counts)

            # add to document counts
            self.add_to_doc_counts(doc_counts, sentence_counts)

            # recover the single counts as list
            count_to_save, legend = self.get_sentence_counts_to_save(sentence_counts)

            # keep counts to write them at the end
            counts_to_save.append(count_to_save)

        # using numpy to sum up partial counts: counts of a whole document
        arr = np.array(counts_to_save)
        sum_up = np.sum(arr, axis=0)

        # save document sentences info if necessary (no floating point here)
        if self.save_partials:
            ext = "caevo_counts"
            self.create_dir(os.path.join(self.root_salidas_CvFE, self.DIR_SINGLE, file_sub_path))
            file_stats_out = os.path.join(self.root_salidas_CvFE, self.DIR_SINGLE, file_sub_path, f"{doc_name}.{ext}")

            np.savetxt(file_stats_out, arr, newline="\n", fmt='%i', header=" ".join(legend), comments="")

        doc_counts["c_tlinks"] = 0
        for tlink_relation_type in self.tlink_relation_types:
            num = self.conta_tag(root, self.ns + self.Tags.TLINK.value, "relation", tlink_relation_type)
            doc_counts[tlink_relation_type] = num
            doc_counts["c_tlinks"] += num

        self.getProportions(doc_counts, sum_up, legend)

        count_to_save, legend = self.get_sentence_counts_to_save(doc_counts)

        return count_to_save, legend

    def conteo_sentence(self, entry):
        sentence_counts = {}

        sentence_counts["c_events"] = 0
        sentence_counts["c_timexes"] = 0

        sentence_counts["c_words"] = len(entry.findall(".//" + self.ns + self.Tags.TOKEN.value))

        for event_type in self.events_types:
            sentence_counts[event_type] = self.conta_tag(entry, self.ns + self.Tags.EVENT.value, "class", event_type)
            sentence_counts["c_events"] += sentence_counts[event_type]

        for timex_type in self.timex_types:
            sentence_counts[timex_type] = self.conta_tag(entry, self.ns + self.Tags.TIMEX.value, "type", timex_type)
            sentence_counts["c_timexes"] += sentence_counts[timex_type]

        dependencies = entry.find(self.ns + self.Tags.DEPS.value).text

        # subject_type
        sentence_counts["c_nsubj"] = dependencies.count("nsubj" + "(") if dependencies else 0
        sentence_counts["c_csubj"] = dependencies.count("csubj" + "(") if dependencies else 0
        sentence_counts["c_xsubj"] = dependencies.count("xsubj" + "(") if dependencies else 0

        # object
        sentence_counts["c_dobj"] = dependencies.count("dobj" + "(") if dependencies else 0

        # phrase_type
        parsing = entry.find(self.ns + self.Tags.PARSE.value).text
        sentence_counts["c_NP"] = parsing.count("(" + "NP") if parsing else 0
        sentence_counts["c_NNP"] = parsing.count("(" + "NNP") if parsing else 0
        sentence_counts["c_VP"] = parsing.count("(" + "VP") if parsing else 0

        return sentence_counts

    def conta_tag(self, root, tag, attrib, attrib_value):
        """
        Counts the occurrences of a tag with a given attribute
         e.g. tag "event" with tag "class" value "OCURRENCE".
        """
        elems = root.findall(".//" + tag + "/[@" + attrib + "='" + attrib_value + "']")

        return len(elems)

    def set_sentence_indicators(self, sentence_counts):
        """
        Get existence indicator of element in sentence
        :param sentence_counts:
        :return:
        """
        elements = list(sentence_counts.keys())

        for element in elements:
            sentence_counts["E_" + element] = 1 if sentence_counts[element] else 0

    def getProportions(self, doc_counts, sum_of_values, legend):
        # considering elements/sentence relationship

        doc_counts["wordsXsent"] = sum_of_values[legend.index("c_words")] / doc_counts["c_sentences"]
        doc_counts["eventsXsent"] = sum_of_values[legend.index("c_events")] / doc_counts["c_sentences"]
        doc_counts["timexXsent"] = sum_of_values[legend.index("c_timexes")] / doc_counts["c_sentences"]
        doc_counts["NPXsent"] = sum_of_values[legend.index("c_NP")] / doc_counts["c_sentences"]
        doc_counts["VPXsent"] = sum_of_values[legend.index("c_VP")] / doc_counts["c_sentences"]
        doc_counts["NNPXsent"] = sum_of_values[legend.index("c_NNP")] / doc_counts["c_sentences"]

        doc_counts["occuXtot_even"] = sum_of_values[legend.index("OCCURRENCE")] / sum_of_values[
            legend.index("c_events")] if sum_of_values[legend.index("c_events")] else 0
        doc_counts["percXtot_even"] = sum_of_values[legend.index("PERCEPTION")] / sum_of_values[
            legend.index("c_events")] if sum_of_values[legend.index("c_events")] else 0
        doc_counts["repoXtot_even"] = sum_of_values[legend.index("REPORTING")] / sum_of_values[
            legend.index("c_events")] if sum_of_values[legend.index("c_events")] else 0

        doc_counts["prop_E_event"] = sum_of_values[legend.index("E_c_events")] / doc_counts["c_sentences"]
        doc_counts["prop_E_timex"] = sum_of_values[legend.index("E_c_timexes")] / doc_counts["c_sentences"]

        doc_counts["prop_E_occu"] = sum_of_values[legend.index("E_OCCURRENCE")] / doc_counts["c_sentences"]
        doc_counts["prop_E_perc"] = sum_of_values[legend.index("E_PERCEPTION")] / doc_counts["c_sentences"]
        doc_counts["prop_E_repo"] = sum_of_values[legend.index("E_REPORTING")] / doc_counts["c_sentences"]

        return doc_counts

    def add_to_doc_counts(self, doc_counts, sentence_counts):
        elements = list(sentence_counts.keys())

        for element in elements:
            if element not in doc_counts.keys():
                doc_counts[element] = 0
            doc_counts[element] += sentence_counts[element]

    def get_sentence_counts_to_save(self, sentence_counts):
        out = []
        out_legend = []

        for key, item in sentence_counts.items():
            out_legend.append(key)
            out.append(item)

        return out, out_legend

    ##############     SECT: UTILES              ############################

    def parseXMLDoc(self, file_name):
        tree = ET.parse(file_name)
        return tree.getroot()

    def get_out_path(self, file_path, corpus_name, ext, new_file_name=""):
        file_ = file_path.replace(corpus_name, corpus_name + "/stats", 1)

        if new_file_name:
            file_ = os.path.dirname(file_) + "/" + new_file_name

        self.create_dir(os.path.dirname(file_))
        return file_ + "." + ext

    def save_string(self, cadena, file_name):
        with open(file_name, 'w') as the_file:
            the_file.write(cadena)

    def create_dir(self, pathOut):
        if not os.path.exists(pathOut):
            os.makedirs(pathOut)

    def _get_kwarg(kwargs, key, default):
        if key in kwargs:
            arg = kwargs[key]
            del kwargs[key]
        else:
            arg = default
        return arg

    def from_files_to_excel_sheets(self, excelFileName):
        data = []
        fnames = []

        for dirName, subdirList, fileList in os.walk(os.path.join(self.dir_xml_path, "general_stats")):

            for fname in fileList:
                dtype = self.get_dtype_dict_for_generalstats()

                df = pd.read_csv(os.path.join(dirName, fname), delimiter=":")
                df.astype(dtype)

                data.append(df)
                fnames.append(fname.replace(".txt", ""))

        with pd.ExcelWriter(excelFileName + '.xlsx') as writer:
            for df, fname in zip(data, fnames):
                df.to_excel(writer, sheet_name=fname, index=False)

    def dfs_to_excel_sheets(self, array_dfs, names, excelFileName):

        with pd.ExcelWriter(excelFileName + '.xlsx') as writer:
            for df, fname in zip(array_dfs, names):
                df.to_excel(writer, sheet_name=fname, index=False)

    def get_general_file_out(self):
        general_file_out = os.path.join(self.root_salidas_CvFE, f"{self.nomCorpus}_caevo_features.csv")

        return general_file_out

    def get_dtype_dict_for_generalstats(self):
        dataKeys = ['file_name', 'c_sentences', 'c_events', 'c_timexes', 'c_words', 'ASPECTUAL',
                    'I_ACTION', 'I_STATE', 'OCCURRENCE', 'PERCEPTION', 'REPORTING', 'STATE',
                    'DATE', 'TIME', 'DURATION', 'SET', 'c_nsubj', 'c_csubj', 'c_xsubj', 'c_dobj',
                    'c_NP', 'c_NNP', 'c_VP', 'E_c_events', 'E_c_timexes', 'E_c_words', 'E_ASPECTUAL',
                    'E_I_ACTION', 'E_I_STATE', 'E_OCCURRENCE', 'E_PERCEPTION',
                    'E_REPORTING', 'E_STATE', 'E_DATE', 'E_TIME', 'E_DURATION', 'E_SET',
                    'E_nsubj', 'E_csubj', 'E_xsubj', 'E_dobj', 'E_NP', 'E_NNP', 'E_VP',
                    'c_tlinks', 'BEFORE', 'AFTER', 'IBEFORE', 'IAFTER', 'INCLUDES',
                    'IS_INCLUDED', 'BEGINS', 'BEGUN_BY', 'ENDS', 'ENDED_BY', 'SIMULTANEOUS',
                    'NONE', 'VAGUE', 'UNKNOWN', 'OVERLAP', 'BEFORE_OR_OVERLAP',
                    'OVERLAP_OR_AFTER', 'wordsXsent', 'eventsXsent', 'timexXsent',
                    'NPXsent', 'VPXsent', 'NNPXsent', 'occuXtot_even', 'percXtot_even',
                    'repoXtot_even', 'prop_E_event', 'prop_E_timex', 'prop_E_occu',
                    'prop_E_repo']
        proportions = ['wordsXsent', 'eventsXsent', 'timexXsent',
                       'NPXsent', 'VPXsent', 'NNPXsent', 'occuXtot_even', 'percXtot_even',
                       'repoXtot_even', 'prop_E_event', 'prop_E_timex', 'prop_E_occu',
                       'prop_E_repo']
        dict = {}

        isString = ['file_name']
        isFloat = ['wordsXsent', 'eventsXsent', 'timexXsent',
                   'NPXsent', 'VPXsent', 'NNPXsent', 'occuXtot_even', 'percXtot_even',
                   'repoXtot_even', 'prop_E_event', 'prop_E_timex', 'prop_E_occu',
                   'prop_E_repo']

        for elem in dataKeys:
            if elem in isString:
                dict[elem] = str
            elif elem in isFloat:
                dict[elem] = float
            else:
                dict[elem] = int

        return dict

    #############     SECT: EVENTS              ############################

    def get_lemma_events_per_caevo_cat_genre(self, dict_genre_types, caevo_corpus_files_dir):
        for dirName, subdirList, fileList in os.walk(caevo_corpus_files_dir):

            for fname in fileList:
                if fname.endswith(".xml"):
                    file_path = os.path.join(dirName, fname)
                    root = self.parseXMLDoc(file_path)
                    files = [elem for elem in root.iter(self.ns + self.Tags.FILE.value)]

                    for file_node in files:
                        doc_name = file_node.get("name")
                        self.get_events_lists(file_node, dict_genre_types)

    def get_events_lists(self, file_node, events_type_list_dict):
        for event in file_node.iter(self.ns + "event"):
            lema = self.get_lemma(event.get("string"), "v")
            if lema not in events_type_list_dict[event.get("class")].keys():
                events_type_list_dict[event.get("class")][lema] = 0
            events_type_list_dict[event.get("class")][lema] += 1

    def get_lemma(self, word, tipe_pos):
        wnl = nltk.WordNetLemmatizer()
        lemma = wnl.lemmatize(word.strip().lower(), pos=tipe_pos)
        return lemma

    def get_sentences_without_events_by_corpus(self, list_of_examples, caevo_files_dir):

        tipes = ["utterance", "question", "exclamation", "other"]
        for dirName, subdirList, fileList in os.walk(caevo_files_dir):
            for fname in fileList:
                if fname.endswith(".info.xml"):
                    file_path = os.path.join(dirName, fname)

                    root = self.parseXMLDoc(file_path)

                    files = [elem for elem in root.iter(self.ns + self.Tags.FILE.value)]
                    for file_node in files:
                        doc_name = file_node.get("name")

                        entries = [elem for elem in file_node.iter(self.ns + self.Tags.ENTRY.value)]

                        for entry in entries:
                            events = entry.find(self.ns + self.Tags.EVENTS.value)

                            if len(events) == 0:

                                features = ['has_not_VB', 'has_ROOT_HAVE', 'has_ROOT_BE', 'has_ROOT_DO', 'other_root',
                                            'has_COP', 'has_colon', 'is_utterance', 'is_question', 'is_exclamation',
                                            'is_other']
                                dict_to_keep = {x: False for x in features}

                                dict_to_keep["file"] = doc_name

                                sentence = entry.find(self.ns + self.Tags.SENTENCE.value).text
                                dict_to_keep["sentence"] = sentence

                                stopwords = ['!', '%', '.', '+', '?', '-', '|', ';', "'", '_', '&', ',', '@', '`', ':',
                                             '~', '$', '\'\'', "\"", "``"]
                                dict_to_keep["length_in_words"] = len(
                                    [x for x in sentence.split(" ") if x not in stopwords])

                                tree = entry.find(self.ns + self.Tags.DEPS.value)
                                parse = entry.find(self.ns + self.Tags.PARSE.value).text
                                dict_to_keep["has_not_VB"] = "(VB" not in parse

                                if tree.text != None:

                                    dependence_tree = tree.text.split("\n")
                                    raw_root = [x for x in dependence_tree if x.lower().startswith("root")]
                                    if len(raw_root) > 0:
                                        root_word = raw_root[0].split(",")[1].split("-")[0]
                                        lemma_root = self.get_lemma(root_word, 'v')

                                        dict_to_keep["has_ROOT_HAVE"] = lemma_root == "have"
                                        dict_to_keep["has_ROOT_BE"] = lemma_root == "be"
                                        dict_to_keep["has_ROOT_DO"] = lemma_root == "do"
                                        dict_to_keep["other_root"] = not (
                                                dict_to_keep["has_ROOT_HAVE"] or dict_to_keep["has_ROOT_BE"] or
                                                dict_to_keep["has_ROOT_DO"])
                                    # verbos copulativos "be,appear,become,reamin,resemble,seem,stay"
                                    dict_to_keep["has_COP"] = len(
                                        [x for x in dependence_tree if x.startswith("cop")]) > 0

                                dict_to_keep["has_colon"] = ":" in sentence

                                quotes = ["\'\'", "\"", "``"]
                                dict_to_keep["is_utterance"] = len([x for x in quotes if x in sentence]) > 0
                                dict_to_keep["is_question"] = "?" in str(sentence)
                                dict_to_keep["is_exclamation"] = "!" in str(sentence)
                                dict_to_keep["is_other"] = not (
                                        dict_to_keep["is_utterance"] or dict_to_keep["is_question"] or dict_to_keep[
                                    "is_exclamation"])
                                list_of_examples.append(dict_to_keep)

    #############     SECT: DMs              ############################
    def get_DMs_words(self, dict_genre_types, genre, caevo_files_dir, DM_df, DM_list):

        dms_list = []

        for dirName, subdirList, fileList in os.walk(caevo_files_dir):
            for fname in fileList:

                if fname.endswith(".info.xml"):
                    file_path = os.path.join(dirName, fname)
                    root = self.parseXMLDoc(file_path)
                    files = [elem for elem in root.iter(self.ns + self.Tags.FILE.value)]
                    for file_node in files:

                        entries = [elem for elem in file_node.iter(self.ns + self.Tags.ENTRY.value)]

                        for entry in entries:
                            features = "genre,struc_meaning,sem_meaning,text".split(",")

                            aux = {}
                            sentence = entry.find(self.ns + self.Tags.SENTENCE.value).text.lower()
                            sentence = sentence.replace(" 's", "'s")

                            DMs = [re.sub(rf"[\.|,| |:|;]$", "", x) for x in np.concatenate(
                                [re.findall(rf"{x}[\.|,| |:|;]", sentence, flags=re.IGNORECASE) for x in DM_list])]

                            for dm in DMs:
                                aux["genre"] = genre
                                aux["text"] = dm
                                aux["struc_meaning"] = DM_df[DM_df.English == dm].iloc[0].structural
                                aux["sem_meaning"] = DM_df[DM_df.English == dm].iloc[0].semantic

                                dms_list.append(aux)

        df_genre = pd.DataFrame(dms_list)
        df_genre = df_genre.groupby(df_genre.columns.tolist()).size().reset_index().rename(columns={0: 'counts'})

        absent = [x for x in DM_df.English.tolist() if x not in df_genre["text"].unique()]
        for elem in absent:
            aux = {}
            aux["genre"] = genre
            aux["text"] = elem
            aux["struc_meaning"] = DM_df[DM_df.English == elem].iloc[0].structural
            aux["sem_meaning"] = DM_df[DM_df.English == dm].iloc[0].semantic
            aux["counts"] = 0
            df_genre = pd.concat([df_genre, pd.DataFrame([aux])])
        dict_genre_types[genre] = df_genre

    #############     SECT: TIMEX              ############################
    def get_timex_words(self, dict_genre_types, genre, caevo_files_dir):
        timex_list = []
        len_inicial = 0

        for dirName, subdirList, fileList in os.walk(caevo_files_dir):

            for fname in fileList:
                if fname.endswith(".xml"):
                    file_path = os.path.join(dirName, fname)
                    root = self.parseXMLDoc(file_path)
                    files = [elem for elem in root.iter(self.ns + self.Tags.FILE.value)]

                    for file_node in files:
                        for timex in file_node.iter(self.ns + "timex"):
                            aux = {}
                            len_inicial += 1

                            aux["text"] = timex.get("text")
                            aux["type"] = timex.get("type")
                            aux["genre"] = genre
                            timex_list.append(aux)

        df_genre = pd.DataFrame(timex_list)
        df_genre = df_genre.groupby(df_genre.columns.tolist()).size().reset_index().rename(columns={0: 'counts'})
        dict_genre_types[genre] = df_genre

    def plot_a_dict_counts(self, dict_to_plot, fileName, need_sorting=False):

        matplotlib.use('agg')
        if need_sorting:
            import operator
            sorted_x = sorted(dict_to_plot.items(), key=operator.itemgetter(1), reverse=True)
            freqs2 = [dict_to_plot.freq(sample) for sample, _ in sorted_x]

        samples = [item for item, _ in dict_to_plot.items()]

        pylab.xlabel("Samples")
        pylab.xticks(range(len(samples)), [str(s) for s in samples], rotation=90)
        pylab.grid(True, color="silver")

        pylab.plot(freqs2)
        pylab.savefig(fileName)

    def dispersion_plot_from_nltk(self, text, ignore_case, words, fileName):
        try:
            from matplotlib import pylab
        except ImportError:
            raise ValueError('The plot function requires matplotlib to be installed.'
                             'See http://matplotlib.org/')
        text = list(text)
        words.reverse()

        if ignore_case:
            words_to_comp = list(map(str.lower, words))
            text_to_comp = list(map(str.lower, text))
        else:
            words_to_comp = words
            text_to_comp = text

        points = [(x, y) for x in range(len(text_to_comp))
                  for y in range(len(words_to_comp))
                  if text_to_comp[x] == words_to_comp[y]]
        if points:
            x, y = list(zip(*points))
        else:
            x = y = ()
        pylab.plot(x, y, "b|", scalex=.1)
        pylab.yticks(list(range(len(words))), words, color="b")
        pylab.ylim(-1, len(words))
        pylab.xlabel("Word Offset")
        pylab.savefig(fileName)

    def plot_from_nltk_save(self, freqList, *args, **kwargs):

        try:
            from matplotlib import pylab
        except ImportError:
            raise ValueError('The plot function requires matplotlib to be installed.'
                             'See http://matplotlib.org/')
        if len(args) == 0:
            args = [len(freqList)]
        samples = [item for item, _ in self.most_common(*args)]

        cumulative = self._get_kwarg(kwargs, 'cumulative', False)
        if cumulative:
            freqs = list(self._cumulative_frequencies(samples))
            ylabel = "Cumulative Counts"
        else:
            freqs = [self[sample] for sample in samples]
            ylabel = "Counts"
        # percents = [f * 100 for f in freqs]  only in ProbDist?

        pylab.grid(True, color="silver")
        if not "linewidth" in kwargs:
            kwargs["linewidth"] = 2
        if "title" in kwargs:
            pylab.title(kwargs["title"])
            del kwargs["title"]
        pylab.plot(freqs, **kwargs)
        pylab.xticks(range(len(samples)), [text_type(s) for s in samples], rotation=90)
        pylab.xlabel("Samples")
        pylab.ylabel(ylabel)
        pylab.savefig("pajarito.png")

    def get_numMax_events_per_type_genre(self, df_gral, numMax):
        new_df_list = []
        events_types = ["ASPECTUAL", "I_ACTION", "I_STATE", "OCCURRENCE", "PERCEPTION", "REPORTING", "STATE"]
        for genre in corpora_genres_train.keys():
            df_gen = df_gral[df_gral["genre"] == genre]
            list_menores = []
            for ev_type in events_types:
                df_aux = df_gen[df_gen["type_event"] == ev_type]
                limite = min(len(df_aux), numMax)
                list_menores.append(df_aux[:limite])
            new_df_list.append(pd.concat(list_menores))

        df_reduced = pd.concat(new_df_list)

        return df_reduced
