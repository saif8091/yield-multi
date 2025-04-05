import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_set_size_21 = 6
test_set_split_22 = 0.3
rand_state = 42

train_pltn_2021, test_pltn_2021 = train_test_split(np.arange(1,19), train_size=train_set_size_21, random_state=rand_state)
train_pltn_2022, test_pltn_2022 = train_test_split(np.arange(1,89), test_size=test_set_split_22, random_state=rand_state)