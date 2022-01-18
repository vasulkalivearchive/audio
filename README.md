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
|argument|description|type|default|
|---|---|---|---|
|`--video_path`|path to video or videos for tagging|str|"path/to/videos/"|
|`--model_path`|path to pre-trained audio tagging model|str|"pretrained/CNN14_Vasulka_1s.pth"|
|`--tag_names`|tag names used by the model|str|"pretrained/tag_names.txt"|
|`--output_path`|output path for predicitions|str|"output/"|
|`--gpu_predict`|if True, tagging runs on GPU|bool|False|
|`--gpu_encode`|if True, video encoding runs on GPU|bool|False|
|`--save_video`|if True, annotations are rendered to video|bool|False|
|`--video_bitrate`|if None, bitrate is set to video height * 4500|int|None|
|`--plot_predict`|if True, prediction plot is saved to .png|bool|False|
|`--skip_videos`|list of videos to skip in video_path|str|[]|

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
