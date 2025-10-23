import pandas as pd
from torch import randperm
import os
import pickle
IMGS_PATH = "../datasets/Text2Img/imgs"
CSV_PATH = "../datasets/Text2Img/results.csv"


# df = pd.read_csv(CSV_PATH, sep="|")
# df["image_name"] = df["image_name"].apply(lambda x: f"{IMGS_PATH}/{x}" if IMGS_PATH not in x else x)
# train_amount = int(df.shape[0] * 0.75)

# arr = df.to_numpy()
# arr = arr[randperm(df.shape[0])]
# df = pd.DataFrame(arr, columns=["filepath", "comment_number", "comment"])

# train = df.loc[:train_amount]
# val = df.loc[train_amount:]

# train.to_csv(f"train.csv", index_label=False)
# val.to_csv(f"val.csv", index_label=False)

imgs = [f"{IMGS_PATH}/{i}" for i in os.listdir(IMGS_PATH)]
with open("metadata.pkl", 'wb') as f:
    data = {
        "imgs_srcs": imgs
    }

    pickle.dump(data, f)