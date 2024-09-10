import os.path

import yt_dlp
import json
import pprint

def check_for_cache_file():
    file_name = yt_channel + ".json"
    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            info_json = json.load(json_file)
            return info_json
    return None


download_location = "../videos/"

yt_channel = "jomezpro/search?query=mcbeth"
# yt_channel = "batterybulletin"

yt_dlp_options = {
    'format': 'bestvideo+bestaudio',  # Download the best audio and video, cannot download them as one.
    'merge-output-format': 'mkv',  # merge the audio and video together
    "download-archive": download_location,  # won't download any videos that have already been downloaded
    "config-location": "yt-dlp.conf",
    "username": "oauth2",
    "password": '',
}
info = check_for_cache_file()
downloader = yt_dlp.YoutubeDL(yt_dlp_options)
if not info:
    print("Cached info not found. Downloading Info")
    # Download the list of videos from channel or playlist
    info = downloader.extract_info("https://www.youtube.com/@" + yt_channel, download=False)
else:
    print("Using cached info file")

# List all keys if needed.
# for key, value in info.items():
# print(key)
# if key == "title":
#     print(f"{key}: {value}")

with open(yt_channel + ".json", "w") as jsono_file:
    json_object = json.dumps(info)
    jsono_file.write(json_object)

# for video in info["entries"]:
#     if "anode" in video["title"]:
#         downloader.download(video["original_url"])
