## yt-dlp
- .\yt-dlp.exe  -f 'bestvideo+bestaudio' --merge-output-format mkv "https://www.youtube.com/watch?v=CS6CPh-KbtI"
- yt-dlp  -f 'bestvideo+bestaudio' --merge-output-format mkv "https://www.youtube.com/@JomezPro"
- yt-dlp -w -P "\\192.168.7.243\nas\dwiebold\jomez_videos" -f 'bestvideo+bestaudio' --merge-output-format mkv "https://www.youtube.com/@JomezPro"
- yt-dlp -w ----match-title "(2022|MPO)" -P "\\192.168.7.243\nas\dwiebold\jomez_videos" -f 'bestvideo+bestaudio' --merge-output-format mkv "https://www.youtube.com/@JomezPro"
#### If you plan to use pycharm you'll need to download ffmpeg and manually add the exe to 
- .venv/Scripts/ffmpeg.exe
# wsl2 disk resize
- `wsl --shutdown`
- `diskpart`
# open window Diskpart
- `select vdisk file="C:\Users\dwieb\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"`
- `attach vdisk readonly`
- `compact vdisk`
- `detach vdisk`
- `exit`

# Package websites
https://github.com/coletdjnz/yt-dlp-youtube-oauth2



# Random notes
- this is wrong but may as well continue
  - `sudo rsync -ah --progress \\192.168.7.243\\nas\\dwiebold\\jomez_videos/ /media/nas/dwiebold/jomez_videos/`