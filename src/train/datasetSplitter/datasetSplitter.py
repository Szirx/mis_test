import pandas as pd
import os
import numpy as np
import configs.config as cfg


def create_df():
    name = []
    for dirname, _, filenames in os.walk(cfg.IMAGE_PATH):
        for filename in filenames:
            name.append(filename.split('.')[0])

    return pd.DataFrame({'id': name}, index=np.arange(0, len(name)))