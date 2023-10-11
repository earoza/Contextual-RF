import pandas as pd

def split_dataframe_rows(df, column_selectors, separator):
    '''
    df = pandas dataframe to split,
    column_selectors = the columns containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    '''

    # we need to keep track of the ordering of the columns
    def _split_list_to_rows(row, row_accumulator, column_selector, row_delimiter):
        split_rows = {}
        max_split = 0
        for column_selector in column_selectors:
            split_row = row[column_selector].split(row_delimiter)
            split_rows[column_selector] = split_row
            if len(split_row) > max_split:
                max_split = len(split_row)

        for i in range(max_split):
            new_row = row.to_dict()
            for column_selector in column_selectors:
                try:
                    new_row[column_selector] = split_rows[column_selector].pop(0)
                except IndexError:
                    new_row[column_selector] = ''
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(_split_list_to_rows, axis=1, args=(new_rows, column_selectors, separator))
    new_df = pd.DataFrame(new_rows, columns=df.columns)
    return new_df

def create_empty_columns(df, columns):
    for col in columns:
        df[col] = None