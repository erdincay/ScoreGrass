import os
import pandas as pd
import sys

__author__ = 'Kern'

if len(sys.argv) < 2:
    raise ValueError("Usage:", sys.argv[0], " Missing some argument to indicate input files")
path = sys.argv[1]

ori_path = os.path.abspath(os.path.join(path, os.pardir))
parent_path = os.path.abspath(os.path.join(ori_path, os.pardir))
print(os.path.isfile(path))

pd.set_option('display.max_columns', 10)
pd.set_option('expand_frame_repr', False)

df = pd.read_excel(path, 'Sheet1', index_col=None)

# print(df)
df = df.dropna(axis=0)
filenames = df['File']
print(filenames)
print(filenames.apply(os.path.isfile))