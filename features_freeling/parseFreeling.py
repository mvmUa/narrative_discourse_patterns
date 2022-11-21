import subprocess
import os
import datetime
import traceback
import nltk
import pandas as pd
import json

Freeling_command = ["/usr/local/bin/analyze"]
"""
Freeling_options
output level --> dep: 
tokenized, sentence-splitted, morphologically analyzed, PoS-tagged, optionally sense-annotated, and dependency-parsed text, 
as output by the second stage (transformation to dependencies and function labelling) of the dependency parser. 
May include also SRL if the statistical parser is used (and SRL is available for the input language).
We also include --ner
"""

Freeling_options = ["-f", "en.cfg", "--ner", "--outlv", "dep", "--output", "conll"]
# with sense
Freeling_options2 = ["-f", "en.cfg", "--sense", "mfs", "--ner", "--outlv", "dep", "--output", "conll", "--dep",
                     "treeler"]
Freeling_options3 = ["-f", "en.cfg", "--input", "freeling", "--ner", "--outlv", "dep", "--output", "conll", "--dep",
                     "treeler"]
# sense desambiguation UKB
Freeling_options4 = ["-f", "en.cfg", "--outlv", "coref", "--ner", "--output", "conll"]

Freeling_options5 = ["-f", "en.cfg", "-s ukb --nec", "--output", "conll"]

f_options = [Freeling_options, Freeling_options2, Freeling_options3]

EXT = ".fre_out"


def parseFreeling_one_doc(input, output, free_opt=Freeling_options5):
    with open(input, 'r', encoding="ISO-8859-1") as fi, open(output, "w") as fo:
        try:
            print("Hora ini: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
            raw = fi.read()
            texto = "\n".join(nltk.sent_tokenize(raw))

            fre_out = subprocess.check_output(Freeling_command + free_opt, input=texto, encoding='utf8')

            # GENERA
            with open(output, "w") as fw:
                fw.write(fre_out)

        except Exception as e:
            print()
            print(e)


def parseFreeling_one_doc_segmenting_before_NO_coref_needed(input, output, free_opt=Freeling_options5):
    with open(input, 'r', encoding="ISO-8859-1") as fi, open(output, "w") as fo:

        print("Hora ini: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
        raw = fi.read()

        texto = nltk.sent_tokenize(raw)
        i = 0
        sents = {}
        for sentence in texto:
            try:

                fre_out1 = subprocess.check_output(Freeling_command + free_opt, input=sentence, encoding='utf8')
                sents[i] = fre_out1
                i += 1

            except Exception as e:
                print(i)
                i += 1
                print(e)

    # GENERA
    with open(output, "w") as fw:
        fw.write("\n".join(list(sents.values())))


def parseFreeling_inColective_out_files(incolective_file, input_type, dirNameOut, file_log_path, free_opt):
    if input_type == "csv":
        df_data = pd.read_csv(incolective_file)
    elif input_type == "json":
        df_data = pd.read_json(incolective_file, lines=True)

    mistake_counter = 0
    for id_row, row in df_data.iterrows():
        print(id_row)
        id_Article = row["Id_Article"]
        for tipe_doc, doc in [(type_doc, row[type_doc]) for type_doc in ['articleBody', 'sentence2', 'sentence1']]:
            try:
                nomFileOut = id_row
                if tipe_doc == "articleBody":
                    nomFileOut = id_Article

                create_dir(os.path.join(dirNameOut, tipe_doc))
                out_file_root = os.path.join(dirNameOut, tipe_doc, f"{nomFileOut}.{EXT}")

                print("Hora ini: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))

                texto = "\n".join(nltk.sent_tokenize(doc))

                fre_out = subprocess.check_output(Freeling_command + free_opt, input=texto, encoding='utf8')
                time = datetime.datetime.now().strftime("%H:%M:%S")

                # GENERA
                with open(out_file_root, "w") as fw:
                    fw.write(fre_out)

            except Exception as e:

                print(f"*****  ruta:{out_file_root}\n")

                with open(file_log_path, 'a') as f:
                    f.write(str(mistake_counter) + "\n")
                    mistake_counter += 1
                    f.write("Hora: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
                    f.write(f"ruta_out: {out_file_root}\n")
                    f.write(str(e) + "\n")
                    f.write(traceback.format_exc() + "\n")
                    f.write("***************************************************\n")


def get_already_computed_from_dir(dir_already):
    list_of_files = [x.replace(".fre_out", "") for x in os.listdir(dir_already)]
    return list_of_files


def parseFreeling_inColective_out_files_with_already_checking(incolective_file, input_type, dirNameOut, file_log_path,
                                                              free_opt):
    if input_type == "csv":
        df_data = pd.read_csv(incolective_file)
    elif input_type == "json":
        df_data = pd.read_json(incolective_file, lines=True)

    mistake_counter = 0
    for id_row, row in df_data.iterrows():
        id_Article = row["Id_Article"]

        dir_already = "sentence"
        already_computed = get_already_computed_from_dir(dir_already)

        if str(id_row) not in already_computed:
            print(id_row)
            for tipe_doc, doc in [(type_doc, row[type_doc]) for type_doc in ['articleBody', 'sentence2', 'sentence1']]:
                # por cada item se genera un archivo
                try:
                    nomFileOut = id_row
                    if tipe_doc == "articleBody":
                        nomFileOut = id_Article

                    create_dir(os.path.join(dirNameOut, tipe_doc))
                    out_file_root = os.path.join(dirNameOut, tipe_doc, f"{nomFileOut}.{EXT}")

                    print("Hora ini: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))

                    # Segmentando con nltk
                    texto = "\n".join(nltk.sent_tokenize(doc))

                    fre_out = subprocess.check_output(Freeling_command + free_opt, input=texto, encoding='utf8')
                    time = datetime.datetime.now().strftime("%H:%M:%S")

                    # GENERA
                    with open(out_file_root, "w") as fw:
                        fw.write(fre_out)

                except Exception as e:

                    print(f"*****  ruta:{out_file_root}\n")

                    with open(file_log_path, 'a') as f:
                        f.write(str(mistake_counter) + "\n")
                        mistake_counter += 1
                        f.write("Hora: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
                        f.write(f"ruta_out: {out_file_root}\n")
                        f.write(str(e) + "\n")
                        f.write(traceback.format_exc() + "\n")
                        f.write("***************************************************\n")


def parseFreeling_with_log(dir_with_texts, dirNameOut, file_log, Freeling_opt):
    create_dir(dirNameOut)

    mistake_counter = 0
    with open(file_log, "a") as fe:
        for dirName, subdirList, fileList in os.walk(dir_with_texts):

            if not os.path.exists(dirNameOut):
                os.makedirs(dirNameOut)

            for fname in fileList:
                doc_out = fname.replace(".txt", EXT) if fname.endswith(".txt") else fname + EXT

                input = os.path.join(dirName, fname)
                output = os.path.join(dirNameOut, doc_out)

                with open(input, "rb") as fi, open(output, "w") as fo:
                    try:
                        raw = fi.read()
                        texto = raw.decode("utf-8", "ignore")
                        texto = "\n".join(nltk.sent_tokenize(texto))
                        # texto = "\n".join(nltk.sent_tokenize(texto)[17:])
                        texto = texto.replace("&quot;", " ")
                        texto = texto.replace("FACILITIESDidn", "FACILITIES Didn")
                        texto2 = "\n".join(nltk.sent_tokenize(texto))
                        s = subprocess.run(Freeling_command + Freeling_opt, input=texto, encoding='utf8', stderr=fe,
                                           stdout=fo)
                        print(input)

                    except Exception as e:

                        print(f"*****  ruta:{input}\n")

                        with open(file_log, 'a') as f:
                            f.write(str(mistake_counter) + "\n")
                            mistake_counter += 1
                            f.write("Hora: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
                            f.write(f"ruta: {input}\n")
                            f.write(str(e) + "\n")
                            f.write(traceback.format_exc() + "\n")
                            f.write("***************************************************\n")


def parseFreeling_in_csv_out_json(in_csv, out_json, log_doc_path, Freeling_opt):
    df_docs = pd.read_csv(in_csv)
    df_docs.columns = []

    out_basename = ""
    if not os.path.exists(out_basename):
        os.makedirs(out_basename)

    numFiles = 0
    mistake_counter = 0

    already_computed_file = pd.read_json(out_json, lines=True)
    already_computed_docs = already_computed_file.ruta.tolist()

    with open(out_json, "a") as doc_json:

        for index, row in df_docs.iterrows():
            data_row = {}
            data_row["ruta"] = row.ruta
            print(row.ruta)
            data_row["doc"] = row.doc

            numFiles += 1
            print(numFiles)

            if row.ruta not in already_computed_docs:

                try:
                    data_row["free_doc"] = subprocess.check_output(Freeling_command + Freeling_opt,
                                                                   input=data_row["headline"], encoding='utf8')
                    doc_json.write(json.dumps(data_row) + "\n")
                except Exception as e:

                    print("*****  ruta: " + data_row["ruta"] + "\n")
                    with open(os.path.join(log_doc_path), 'a') as f:
                        f.write(str(mistake_counter) + "\n")
                        mistake_counter += 1
                        f.write("Hora: " + datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S\n"))
                        f.write("ruta: " + data_row["ruta"] + "\n")
                        f.write(str(e) + "\n")
                        f.write(traceback.format_exc() + "\n")
                        f.write("***************************************************\n")


def create_dir(pathOut):
    if not os.path.exists(pathOut):
        os.makedirs(pathOut)
