# vasulkalivearchive – Audio Tagging
This repository contains a console application used for audio content tagging in videos.

### Running the console application
Example 1 – Tagging just one video
```
python audio_tagging.py --video_path "path/to/video/video.mp4"
```
Example 2 – Tagging multiple videos in one folder
```
python audio_tagging.py --video_path "path/to/videos"
```

### audio_tagging.py arguments
|<sub> argument|<sub> description|<sub> type|<sub>default|
|---|---|---|---|
|<sub> `--video_path` |<sub>path to video or videos for tagging|<sub>str|<sub>"path/to/videos/"|
|<sub>`--model_path`|<sub> path to pre-trained audio tagging model|<sub>str|<sub>"pretrained/CNN14_Vasulka_1s.pth"|
|<sub>`--tag_names`| <sub> tag names used by the model|<sub>str|<sub>"pretrained/tag_names.txt"|
|<sub>`--output_path`|<sub>output path for predicitions|<sub>str|<sub>"output/"|
|<sub>`--gpu_predict`|<sub>if True, tagging runs on GPU|<sub>bool|False|
|<sub>`--gpu_encode`|<sub>if True, video encoding runs on GPU|<sub>bool|False|
|<sub>`--save_video`|<sub>if True, annotations are rendered to video|<sub>bool|False|
|<sub>`--video_bitrate`|<sub>if None, bitrate is set to video height * 4500|<sub>int|None|
|<sub>`--plot_predict`|<sub>if True, prediction plot is saved to .png|<sub>bool|False|
|<sub>`--skip_videos`|<sub>list of videos to skip in video_path|<sub>str|[]|


### The model allows you to tag these categories:

| **Tag name**             | **F1-score** [%] |
|--------------------------|----------|
| Music (acoustic)         | 86.06    |
| Music (electronic)       | 91.15    |
| Music (violin)           | 83.96    |
| Music (vocal)            | 79.58    |
| Noise                    | 93.39    |
| Noise (air)              | 78.55    |
| Noise (car)              | 77.04    |
| Noise (fire)             | 96.55    |
| Noise (water)            | 86.79    |
| Speech                   | 93.39    |

### Dependencies
```
matplotlib==3.3.3
numpy==1.19.5
opencv_python==4.5.1.48
moviepy==1.0.3
torchlibrosa==0.0.4
tqdm==4.61.1
torch==1.5.0+cu101
librosa==0.8.0
```
