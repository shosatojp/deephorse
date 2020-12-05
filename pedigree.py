import pandas as pd
from tqdm import tqdm
from concurrent.futures.process import ProcessPoolExecutor
import os
import numpy as np


def init_ancestors(ancestor_depth):
    '''
    ancestor_pos: 0 ~ 1+(2**ancestor_depth)をヒープに入れたときの位置
    奇数: man
    偶数: woman
    '''
    for ancestor_pos in range(1, 2**(ancestor_depth+1) - 2 + 1):
        df[f'ancestor_{ancestor_pos}'] = -1  # pd.Series(dtype=np.int64)
        df[f'ancestor_{ancestor_pos}'] = df[f'ancestor_{ancestor_pos}'].astype(np.int64)


def get_parents(df, row_index, ancestor_depth, name_rows_map, child_pos, series):
    if child_pos <= 2**(ancestor_depth) - 2:
        father = series[8]
        mother = series[9]

        father_pos = child_pos * 2 + 1
        mother_pos = child_pos * 2 + 2

        for parent, parent_pos in zip([father, mother],
                                      [father_pos, mother_pos]):
            parents = name_rows_map.get(parent)

            if parents and len(parents) == 1:
                # row id
                parent_id = parents[0][0]
                df.at[row_index, f'ancestor_{parent_pos}'] = parent_id
                get_parents(df, row_index, ancestor_depth, name_rows_map, parent_pos, parents[0])

    else:
        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-in', required=True)
    parser.add_argument('--output', '-out', required=True)
    parser.add_argument('--depth', '-d', required=True, type=int)
    args = parser.parse_args()

    # load csv
    print('loading csv')
    df = pd.read_csv(args.input)

    # init ancestor
    ancestor_depth = args.depth
    init_ancestors(ancestor_depth)

    # create name-row map
    print('creating map')
    name_rows_map = {}
    for series in tqdm(df.itertuples(name=None), total=len(df)):
        name_rows_map.setdefault(series[3], []).append(series)

    # resolve ancestors
    print('resolving ancestors')
    for series in tqdm(df.itertuples(name=None), total=len(df)):
        get_parents(df, series[0], ancestor_depth, name_rows_map, 0, series)

    print(df)
    df.to_csv(args.output)
