import os.path
import yt_dlp
import json


def check_for_cache_file():
    info_name = file_name + ".json"
    if os.path.exists(info_name):
        with open(info_name, "r") as json_file:
            info_json = json.load(json_file)
            return info_json
    return None


def download_metadata(url):
    """
    This method can download the metadat for any given channel or playlist.
    Keyword being 'can' because youtube limits the number of api hits. It's easier to use the
    inspect console scripts that will be in the readme.
    :param url: string
    :return: yt_dpl info object/dict
    """
    # Download the list of videos from channel or playlist
    info = downloader.extract_info(url, download=False)

download_location = "../videos/"
file_name = "jomezpro_2022_Las_Vegas_Challenge"
# yt_channel = "jomezpro/search?query=" + file_name
yt_channel = "jomezpro"

# yt_channel = "batterybulletin"

yt_dlp_options = {
    'format': 'bestvideo+bestaudio',  # Download the best audio and video, cannot download them as one.
    'merge-output-format': 'mkv',  # merge the audio and video together
    "download-archive": download_location,  # won't download any videos that have already been downloaded
    "dateafter": 20220101,  # YYYYMMDD
    "datebefore": 20230101,
    # "config-location": "yt-dlp.conf",
    "username": "oauth2",
    "password": '',
}
info = check_for_cache_file()
downloader = yt_dlp.YoutubeDL(yt_dlp_options)
if not info:
    print("Cached info not found. Downloading Info")

else:
    print("Using cached info file")

# List all keys if needed.
# for key, value in info.items():
# print(key)
# if key == "title":
#     print(f"{key}: {value}")

with open(file_name + ".json", "w+") as jsono_file:
    json_object = json.dumps(info)
    jsono_file.write(json_object)

# for video in info["entries"]:
#     if "anode" in video["title"]:
#         downloader.download(video["original_url"])

for video in info["entries"]:
    if "aderhold" in video["title"].lower():
        print(video["title"])