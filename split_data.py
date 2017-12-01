import pandas as pd
import numpy as np

PATH_TO_ORIGINAL_DATA = '/home/ljm/dataset/yoochoose/'
PATH_TO_SAMPLED_DATA = '/home/ljm/dataset/yoochoose/sampled/'
data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'yoochoose-clicks.dat', sep=',', header=None)
data = data.sample(frac=0.1,random_state=0,axis=0)
data.to_csv(PATH_TO_SAMPLED_DATA + 'yoochoose-clicks.dat', sep=',', index=False, header=None)
