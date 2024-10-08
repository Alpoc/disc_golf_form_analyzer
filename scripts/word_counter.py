import glob
import pandas as pd
import math
import numpy
import re

files = glob.glob("../channel_csv/*.csv")

# print(files)

combined_df = pd.DataFrame()
for file in files:
    print(file)
    df = pd.read_csv(file)
    combined_df = pd.concat([combined_df, df])

# print(combined_df.values)
print(combined_df.shape)

counts = {}

for row in combined_df.iterrows():
    columns = list(row[1])
    columns = [x for x in columns if str(x) != 'nan']
    video_name = columns[-2]
    video_url = columns[-1]

    regex = re.compile('[^a-zA-Z]')
    # video_name = regex.sub('', video_name)
    words = video_name.split()
    # words = [regex.sub('', word) for word in words]
    for word in words:
        word = word.replace(",", "")
        # if "MPO" not in words:
        #     if "McBeth" in words:
        #         print(words)
        #     continue
        if counts.get(word) is None:
            counts[word] = 1
        else:
            counts[word] += 1
counts = dict(sorted(counts.items(), key=lambda item: item[1]))


remove_list = ["Jomez", "|", "Open", "Disc", "Round", "Golf", 'Part', '-', '2', '1', '9', 'F9', 'B9', "Gatekeeper",
               "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "The", "MPO", "FPO", "Media",
               "at", "3", "Challenge", "Final", "the", "FINAL", "Championship", "Championships", "Pro", "RD", "DISC",
               "&", "LEAD", "GOLF", "PDGA", "Practice", "Worlds", "Cup", "Card", "State", "FINALF9", "FINALB9"]
for remove_word in remove_list:
    if counts.get(remove_word, None):
        counts.pop(remove_word, None)
print(counts)

# print(counts["Gannon"])