
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


# sapiens_lite install
- instructions here
  - https://github.com/facebookresearch/sapiens/blob/main/lite/README.md
- grab checkpoint not specified in readme
  - https://huggingface.co/facebook/sapiens-pose-bbox-detector
- Because pytorch12.1 is used we need same version of cuda. Install it in conda to keep compatibility similar to venv
- `conda install nvidia/label/cuda-12.1.0::cuda-toolkit`
- mmcv can take a long time to install. mmcv-lite for non cuda
  - Maybe not needed `pip install mmcv`


### Checkpoints
- download the required checkpoints depth, pose, and  detector
- Check the corresponding demo torchscript file for the MODEL line containing the file structure.
  - example `sapiens_lite_host/torchscript/depth/checkpoints/sapiens_0.3b/sapiens_0.3b_render_people_epoch_100_torchscript.pt2`

### What needs what
- pose can be ran without det but the results suffer
- depth requires seg