'''
From https://stackoverflow.com/questions/17834995/how-to-convert-opendocument-spreadsheets-to-a-pandas-dataframe
'''
import ezodf
import pandas as pd


def get_Doc(file_name):
    return ezodf.opendoc(file_name)


def printDocInfo(doc):
    print("Spreadsheet contains %d sheet(s)." % len(doc.sheets))
    for sheet in doc.sheets:
        print("-" * 40)
        print("   Sheet name : '%s'" % sheet.name)
        print("Size of Sheet : (rows=%d, cols=%d)" % (sheet.nrows(), sheet.ncols()))


def doc2Dfs(doc):
    dfs = {}

    for sheet in doc.sheets:
        dfs[sheet.name] = sheet2Df(sheet)

    return dfs


def sheet2Df(sheet):
    df_dict = {}
    for i, row in enumerate(sheet.rows()):
        # row is a list of cells
        # assume the header is on the first row
        if i == 0:
            # columns as lists in a dictionary
            df_dict = {cell.value: [] for cell in row}
            # create index for the column headers
            col_index = {j: cell.value for j, cell in enumerate(row)}
            continue
        for j, cell in enumerate(row):
            # use header instead of column index
            df_dict[col_index[j]].append(cell.value)
    # and convert to a DataFrame
    df = pd.DataFrame(df_dict)
    return df
