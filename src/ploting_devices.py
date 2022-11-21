import math
import numpy as np
import matplotlib.pyplot as plt


class Ploting_devices:
    """
    Class that allows plotting and saving different types of diagrams for reflecting df
    """

    @staticmethod
    def stack_df(df, title="", name_file="", format=""):
        # tamaño de la figura
        fig_width_cm = 12.5  # Get this from LaTeX using width
        fig_width_cm = 14  # Get this from LaTeX using width
        golden_mean = (math.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_width = Ploting_devices.cm2inch(fig_width_cm)  # width in inches
        fig_height = fig_width * golden_mean  # height in inches
        fig_size = [fig_width, fig_height]

        colors1 = "#f6d186,#fcf7bb,#cbe2b0,#f19292,#f4e04d,#8db1ab".replace(" ", "#").split(",")

        df.rename(columns={x: x[0:3] for x in df.columns}, inplace=True)
        df.plot.bar(stacked=True, color=colors1, figsize=fig_size, title=title, rot=0)

        plt.legend(loc='upper left', bbox_to_anchor=(0.91, 1))
        plt.savefig(f"{name_file}.{format}", format="eps")
        if name_file != "":
            plt.savefig(f"{name_file}.{format}", format=format)
            plt.savefig(f"{name_file}.jpg", format="jpg")

    @staticmethod
    def pintar(df_dict):
        plt.hist(df_dict["duc"]["vform_present"], bins=20, label="duc", histtype="step")
        plt.hist(df_dict["tales"]["vform_present"], bins=20, label="tales", histtype="step")
        plt.hist(df_dict["sfu"]["vform_present"], bins=20, label="sfu", histtype="step")
        plt.show()

    @staticmethod
    def grouped_barplot(df, path_nom_graph, ext):
        # Each group represents a category for the three genres
        # Make the plot
        colors = ['#b5ffb9', '#a3acff', '#f9bc86', "#f5b041", "#f7dc6f"]

        df.T.plot(kind='bar', rot=0, color=colors)

        plt.savefig(path_nom_graph + "." + ext)
        plt.savefig(path_nom_graph + "." + "jpg")

    @staticmethod
    def grouped_barplot_demo():
        # set width of bar
        barWidth = 0.25

        # set height of bar
        bars1 = [12, 30, 1, 8, 22]  # bars1 sería el género, un valor para cada columna DATE, SET,...
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

    @staticmethod
    def cm2inch(value):
        return value / 2.54
