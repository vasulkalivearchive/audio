# vasulkalivearchive – Audio Tagging
This repository contains a console application for tagging audio tracks of videos using a [CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn) convolutional neural network trained on a dataset of works by Steina and Woody Vasulka.  
The model trained on the Vasulka dataset can be downloaded [here](odkaznamodel).

### The model allows you to tag these categories:

| **Tag name**             | **F1-score** [%] |
|--------------------------|----------|
| Acoustic music        | 86.06    |
| Electronic music       | 91.15    |
| Violin playing          | 83.96    |
| Vocal Music            | 79.58    |
| Noise                    | 93.39    |
| Air (noise)              | 78.55    |
| Car (noise)              | 77.04    |
| Fire (noise)             | 96.55    |
| Water (noise)            | 86.79    |
| Speech                   | 93.39    |


### Running the console application
Example 1 – Tagging one video
```
python audio_tagging.py --video_path "path/to/video/video.mp4"
```
Example 2 – Tagging multiple videos contained in one folder
```
python audio_tagging.py --video_path "path/to/videos"
```

### audio_tagging.py arguments
|<sub> argument|<sub> description|<sub> type|<sub>default|
|---|---|---|---|
|<sub> `--video_path` |<sub>path to a video or videos for tagging|<sub>str|<sub>"path/to/videos/"|
|<sub>`--model_path`|<sub> path to a pre-trained audio tagging model|<sub>str|<sub>"pretrained/CNN14_Vasulka_1s.pth"|
|<sub>`--tag_names`| <sub> tag names used by the model|<sub>str|<sub>"pretrained/tag_names.txt"|
|<sub>`--output_path`|<sub>output path for predicitions|<sub>str|<sub>"output/"|
|<sub>`--gpu_predict`|<sub>if True, tagging will run on a GPU|<sub>bool|<sub>False|
|<sub>`--gpu_encode`|<sub>if True, the video encoding will run on a GPU|<sub>bool|<sub>False|
|<sub>`--save_video`|<sub>if True, annotations will be rendered to video|<sub>bool|<sub>False|
|<sub>`--video_bitrate`|<sub>if None, bitrate will be set to video height * 4500|<sub>int|<sub>None|
|<sub>`--plot_predict`|<sub>if True, prediction plot will be saved to .png|<sub>bool|<sub>False|
|<sub>`--skip_videos`|<sub>list of videos to skip in video_path|<sub>str|<sub>[]|

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
### Acknowledgements

This repository uses the [CNN14](https://github.com/qiuqiangkong/audioset_tagging_cnn) model proposed in:  
Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.
