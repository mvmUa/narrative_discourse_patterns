import matplotlib.pyplot as plt

matplotlib.use('agg')
import pandas as pd
import os
import numpy as np

"""
https://code.likeagirl.io/an%C3%A1lisis-y-visualizaci%C3%B3n-de-datos-con-pandas-matplotlib-85ee4d7b4cad
"""

ducs = ["duc2002", "duc2003", "duc2004", "duc2006", "duc2007"]
ducs_names = ["DUC_Raw/" + duc + "-preproc" for duc in ducs]

corpora = ["LobosMatosPlain", "SFU", "dormir", "andersen", "ReviewHotelesRoma_TripAdvisorElena"] + ducs_names

# tipos de eventos
event_types = ['ASPECTUAL', 'I_ACTION', 'I_STATE', 'OCCURRENCE', 'PERCEPTION', 'REPORTING', 'STATE']

# ruta del directorio
dir_path = "//salidas/archivos_out"

# ruta de la lista de eventos
event_files_dir = "//salidas/archivos_out"
general_stats_path = "//salidas/archivos_out/general_stats"

corpus = "andersen"
head = ("file_name,#sentences,#tlinks,#events,#timexes,#words,event_type:ASPECTUAL,event_type:I_ACTION,"
        "event_type:I_STATE,event_type:OCCURRENCE,event_type:PERCEPTION,event_type:REPORTING,event_type:STATE,"
        "subject_type:nsubj,subject_type:dsubj,object_type:dobj,phrase_type:NP,phrase_type:NNP,phrase_type:VP,"
        "timex_type:DATE,timex_type:TIME,timex_type:DURATION,timex_type:SET,events/sentece,timexes/sentece,"
        "NP/sentece,VP/sentece,NNP/sentece,occurence/total_events,perception/total_events,reporting/total_events,"
        "tlink_relation:BEFORE,tlink_relation:AFTER,tlink_relation:IBEFORE,tlink_relation:IAFTER,"
        "tlink_relation:INCLUDES,tlink_relation:IS_INCLUDED,tlink_relation:BEGINS,tlink_relation:BEGUN_BY,"
        "tlink_relation:ENDS,tlink_relation:ENDED_BY,tlink_relation:SIMULTANEOUS,tlink_relation:NONE,"
        "tlink_relation:VAGUE,tlink_relation:UNKNOWN,tlink_relation:OVERLAP,tlink_relation:BEFORE_OR_OVERLAP,"
        "tlink_relation:OVERLAP_OR_AFTER")
header = head
header = header.replace("/", "x")
sobra = "event_type:,#,tlink_relation:,phrase_type:,timex_type:,subject_type:,object_type:,total_".split(",")
for elem in sobra:
    header = header.replace(elem, "")
header = header.split(",")

data = pd.read_csv(os.path.join(general_stats_path, corpus + ".txt"), delimiter=":", names=header)


def cm2inch(value):
    return value / 2.54


def percent_stacked_bar_event_types(df):
    # Desde https://python-graph-gallery.com/13-percent-stacked-barplot/
    # Data
    r = [0, 1, 2]

    # plot
    barWidth = 0.85
    names = df.index.tolist()  # ('Tales', 'News', 'Reviews')
    columns = df.columns.tolist()
    tags = [str(x).replace("xcorpEvents", "").lower() for x in columns]

    colors = ['#b5ffb9', '#f9bc86', '#a3acff', "#f5b041", "#f7dc6f"]

    i = 0
    bottom = np.zeros(len(tags))
    for tag, color in zip(columns, colors):
        if i == 0:
            plt.bar(r, df[tag], color=color, edgecolor='white', width=barWidth, label=tags[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            bottom = bottom + np.array(df["ASPxcorpEvents"].values, dtype=float)
            # # Create rest bars
            plt.bar(r, df[tag], bottom=bottom, color=color, edgecolor='white', width=barWidth)
        i += 1

    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("group")

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    # Show graphic
    plt.savefig("stackedBar")


def stackedbarplot(x_data, y_data_list, colors, y_data_names="", x_label="", y_label="", title=""):
    _, ax = plt.subplots()
    # Draw bars, one category at a time
    for i in range(0, len(y_data_list)):
        if i == 0:
            ax.bar(x_data, y_data_list[i], color=colors[i], align='center', label=y_data_names[i])
        else:
            # For each category after the first, the bottom of the
            # bar will be the top of the last category
            ax.bar(x_data, y_data_list[i], color=colors[i], bottom=y_data_list[i - 1], align='center',
                   label=y_data_names[i])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc='upper right')


def percent_stacked_bar_sample():
    # Desde https://python-graph-gallery.com/13-percent-stacked-barplot/

    # Data
    r = [0, 1, 2, 3, 4]
    raw_data = {'greenBars': [20, 1.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15], 'blueBars': [2, 15, 18, 5, 10]}
    df = pd.DataFrame(raw_data)

    # From raw value to percentage
    totals = [i + j + k for i, j, k in zip(df['greenBars'], df['orangeBars'], df['blueBars'])]
    greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
    orangeBars = [i / j * 100 for i, j in zip(df['orangeBars'], totals)]
    blueBars = [i / j * 100 for i, j in zip(df['blueBars'], totals)]

    # plot
    barWidth = 0.85
    names = ('A', 'B', 'C', 'D', 'E')
    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white',
            width=barWidth)

    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("group")

    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="group A")
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="group B")
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white',
            width=barWidth, label="group C")

    # Custom x axis
    plt.xticks(r, names)
    plt.xlabel("group")

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

    # Show graphic
    plt.show()


def grouped_barplot():
    # Desde https://python-graph-gallery.com/11-grouped-barplot/
    # set width of bar
    barWidth = 0.25

    # set height of bar
    bars1 = [12, 30, 1, 8, 22]
    bars2 = [28, 6, 16, 5, 10]
    bars3 = [29, 3, 24, 25, 17]

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Make the plot
    plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='var1')
    plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='var2')
    plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')

    # Add xticks on the middle of the group bars
    plt.xlabel('group', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])

    # Create legend & Show graphic
    plt.legend()
    plt.show()


def carga_datos_caevo_medias(file, csv_info):
    df_dic = {}

    df = pd.read_csv(file, header=None)

    csv_cols = [5, 4, 8, 8, 5, 5, 5, 5, 19, 4, 5, 8, 4, 4, 4]

    i = 0
    dfs = len(csv_info)
    for i in range(0, dfs):
        name = csv_info[i]
        ini_ind = i * 4
        last_col = csv_cols[i]
        header = df.iloc[ini_ind, 0:last_col]
        df_new = pd.DataFrame(df.iloc[ini_ind + 1:ini_ind + 4, 0:last_col])
        df_new.columns = header
        df_new.set_index("Genre", inplace=True)
        df_new = df_new.astype(float)
        if name == "type_event_over_total_event":
            df_new.boxplot()
        df_dic[csv_info[i]] = df_new

    return df_dic


def carga_datos_caevo_medias_2(file):
    import re
    with open(file) as fp:
        lines = [re.sub(",,.*$", "", line)[0:-1] for line in fp.readlines()]

        head_and_desc = [line for line in lines if "#" in line]
        descriptions = [line.split("#")[len(line.split("#")) - 1] for line in head_and_desc]
        headers = [line.replace(",#" + desc, "") for line, desc in zip(head_and_desc, descriptions)]

        df_dict = {}
        for i, desc, head in zip(list(range(0, len(headers))), descriptions, headers):
            ini_item = i * 4 + 1
            data = [line.split(",") for line in lines[ini_item:i * 4 + 4]]
            df = pd.DataFrame(data, columns=head.split(","))
            df.set_index("Genre", inplace=True)
            df = df.astype(float)
            df_dict[desc] = df
        return df_dict
