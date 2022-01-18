import moviepy.editor as mpe
import datetime as dt
import numpy as np

import subprocess
import torch
import json
import cv2
import sys
import os

from pytorch_utils import move_data_to_device
from matplotlib import pyplot as plt
from matplotlib import dates
from math import floor
from librosa import load
from tqdm import tqdm
from time import time


def convert_video(video_path, sr=16000, mono=True, remove_temp=True, out=None):
    dirname, filename = os.path.split(video_path)
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = (cap.get(cv2.CAP_PROP_FPS))
    duration = floor(n_frames / fps)
    if out is not None:
        filename_out = out
    else:
        filename_out = dirname + '/' + os.path.splitext(filename)[0] + '.wav'
    try:
        if mono:
            channels = '1'
        else:
            channels = '2'
        cmd = 'ffmpeg -y -loglevel fatal -i "{}" -ac {} -ar {} -vn {} -f f32le'.format(video_path, channels,
                                                                                       sr, filename_out)
        subprocess.call(cmd, shell=True)
    except IOError:
        sys.exit(1)
    audio_array, _ = load(filename_out, sr=sr)
    if remove_temp:
        os.remove(filename_out)
    return audio_array, duration


def predict_audio(audio_array, model, device, sr=16000, timestep=1, batch_size=1, n_classes=10):
    # get step length in samples
    step = int(timestep * sr)
    # pad array with zeros to an integer number of seconds
    audio_array = np.concatenate([audio_array, np.zeros(step - len(audio_array) % step, dtype=np.float32)])
    last_segment = int((len(audio_array) / step) % batch_size)
    new_length = int(len(audio_array) / step - last_segment)
    # reshape the array by the length of the step
    batch_array = audio_array.reshape((int(len(audio_array) / step), step))
    # create empty array for predictions
    pred_array = np.zeros([(int(len(audio_array) / step)), n_classes])
    # predict in batches
    for i in tqdm(range(0, new_length, batch_size), file=sys.stdout):
        batch = batch_array[i: i + batch_size, :]
        if device == 'cuda':
            batch = move_data_to_device(batch, device)
        tensor_pred = torch.exp((model(torch.tensor(batch)))['clipwise_output'])
        pred = tensor_pred.detach().cpu().numpy()
        pred_array[i: i + batch_size, :] = pred
    # predict the leftover batch if there is one
    if last_segment > 0:
        print('Predicting leftover batch...')
        for j in tqdm(range(0, last_segment), file=sys.stdout):
            batch = move_data_to_device(batch_array[-last_segment + j:, :], device)
            tensor_pred = torch.exp((model(torch.tensor(batch)))['clipwise_output'])
            pred = tensor_pred.detach().cpu().numpy()
            pred_array[-last_segment + j:, :] = pred
    return np.transpose(pred_array)


def render_predict(video_name, pred_array, output_path, video_path, tag_names, max_tags=3,
                   min_prob=0.1, bitrate=4000, gpu_encode=False):
    # set video capture in opencv and get its properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # define name for rendered video
    out_name = os.path.join(output_path, (video_name + '_temp.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # define video temporary video writer
    video_out = cv2.VideoWriter(out_name, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    # define the text properties
    font = {'type': cv2.FONT_HERSHEY_DUPLEX,
            'scale': 0.7,
            'color': (66, 185, 245),
            'thickness': 1}
    # text background alpha (opacity)
    alpha = 0.7
    # text separation factor
    sep = 1.5
    # set to first second
    seconds = 1
    # iterate through video frames using opencv
    for f in tqdm(range(total_frames - 1), unit=' frames', file=sys.stdout):
        # read the frame
        _, frame = cap.read()
        # select top n predictions
        pred_column = list(pred_array[:, seconds - 1])
        print(seconds - 1)
        preds, tags = zip(*sorted(zip(pred_column, tag_names), reverse=True))
        selected_tags, selected_preds = [], []
        for i in range(max_tags):
            if preds[i] > min_prob:
                selected_tags.append(tags[i])
                selected_preds.append(preds[i])
        # if there are some predictions, print them to the frame
        if len(selected_tags) > 0:
            longest_tag = max(selected_tags, key=len)
            # get the longest text
            max_text = ''.join([longest_tag, ': ', '{:.1f}'.format(max(selected_preds) * 100), ' %'])
            text_size, _ = cv2.getTextSize(text=max_text,
                                           fontFace=font['type'],
                                           thickness=font['thickness'],
                                           fontScale=font['scale'])
            # compute appropriate text background dimensions according to the longest text
            rect_width = text_size[0]
            rect_height = int(len(selected_tags) * text_size[1] * sep) + int(text_size[1] * 0.5)
            # create a rectangle text background
            rectangle_bg = frame.copy()
            cv2.rectangle(rectangle_bg, (0, 0), (rect_width, rect_height), (0, 0, 0), -1)
            # add the text background to the current frame
            frame = cv2.addWeighted(rectangle_bg, alpha, frame, 1 - alpha, 0)
            # print tag names and probabilities on top of the rectangle background
            for j, text in enumerate(selected_tags):
                pred = selected_preds[j]
                text_to_print = ''.join([text, ': ', '{:.1f}'.format(pred * 100), ' %'])
                text_coordinates = (0, int((j + 1) * (text_size[1]) * sep))
                cv2.putText(frame, text_to_print, text_coordinates, font['type'], font['scale'],
                            font['color'], font['thickness'], cv2.LINE_AA)
        video_out.write(frame)
        # increment annotating position
        # print(seconds)
        if (cap.get(cv2.CAP_PROP_POS_MSEC)/1000) >= seconds:
            seconds += 1
    video_out.release()
    # create a video clip from the opencv writer
    out_video = mpe.VideoFileClip(out_name)
    # get the original video with audio
    video_waudio = mpe.VideoFileClip(video_path)
    # combine the annotated video with the original audio
    final_video = out_video.set_audio(video_waudio.audio)
    # set the default bitrate
    if bitrate is None:
        bitrate = height * 4500
    out_path = os.path.join(output_path, (video_name + '.mp4'))
    t1 = time()
    if gpu_encode:
        print('Encoding using GPU...')
        final_video.write_videofile(out_path,
                                    audio_codec='libvorbis',
                                    temp_audiofile='./output/temp_audio.ogg',
                                    codec='mpeg4',
                                    bitrate=str(bitrate),
                                    ffmpeg_params=['-vcodec', 'h264_nvenc', '-preset', 'fast'],
                                    logger=None)
    else:
        print('Encoding using CPU...')
        final_video.write_videofile(out_path,
                                    audio_codec='libvorbis',
                                    temp_audiofile='./output/temp_audio.ogg',
                                    codec='mpeg4',
                                    bitrate=str(bitrate),
                                    preset='medium',
                                    logger=None)
    t2 = time()
    print('Encoded in:', '{:.1f}'.format(t2 - t1), 'seconds.')
    if os.path.exists(out_name):
        os.remove(out_name)


def plot_predict(name, pred_array, output_path, tag_names):
    plt.rcParams['figure.figsize'] = (10, 4)  # (15, 3)
    fig, ax = plt.subplots(1)
    x_lims = list(map(dt.datetime.utcfromtimestamp, [0, np.shape(pred_array)[1]]))
    x_lims = dates.date2num(x_lims)
    y_lims = [np.shape(pred_array)[0] - 1, 0]
    im = ax.imshow(pred_array, extent=[x_lims[0], x_lims[1], y_lims[0] + 0.5, y_lims[1] - 0.5],
                   interpolation='none', aspect='auto', cmap='viridis')
    im.set_clim(0, 1)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel('Time (HH:MM:SS)', fontsize=10)
    ax.set_yticks(np.arange(len(tag_names)))
    ax.set_yticklabels(tag_names, fontsize=9)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S'))
    plt.tight_layout()
    fig_name = os.path.join(output_path, (name + '.png'))
    plt.savefig(fig_name)


def pred_json(name, pred_array, output_path, duration, tag_names):
    predict_json = {}
    current_tag = []
    pred_array = pred_array[:, :duration]
    for i in range(len(tag_names)):
        array_row = pred_array[i, :]
        for j in range(len(array_row)):
            # time to msecs
            current_tag.append({"time": float(j * 1000), "prediction": float(array_row[j])})
        predict_json[tag_names[i]] = current_tag
        current_tag = []
    json_path = os.path.join(output_path, (name + '.json'))
    with open(json_path, 'w') as json_file:
        json.dump(predict_json, json_file)
