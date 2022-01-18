from audio_utils import *
import argparse
import os
from audioset_models import TransferCNN14
import sys
import torch


def audio_tagging(args):
    # load audio tags used by the pre-trained model
    try:
        with open(args.tag_names) as f:
            audio_tags = [line.rstrip() for line in f]
    except IOError:
        print('ERROR, file', '\"' + args.tag_names + '\"', 'doesn\'t exist.')
        sys.exit(1)
    # load pre-trained model
    print('Loading pre-trained model...')
    with torch.no_grad():
        model = TransferCNN14(sample_rate=16000, window_size=512, hop_size=128, mel_bins=64,
                              fmin=0, fmax=8000, classes_num=len(audio_tags), freeze_base=False)
        model.load_state_dict(state_dict=torch.load(args.model_path), strict=False)
        if args.gpu_predict:
            device = 'cuda'
        else:
            device = 'cpu'
        model.to(device)
        # set model to evaluation
        model.eval()
    # check if video_path is a directory
    if os.path.isdir(args.video_path):
        files = os.listdir(args.video_path)
    else:
        files = [args.video_path]
    # loop through videos
    for file in files:
        if os.path.isdir(args.video_path):
            path = os.path.join(args.video_path, file)
        else:
            path = file
        video_name = os.path.splitext(os.path.basename(path))[0]
        if file in args.skip_videos:
            continue
        else:
            print('Converting video', path, 'to wav @ 16kHz...')
            try:
                # convert audio to numpy array (one channel, sampling rate = 16 kHz)
                audio_array, duration = convert_video(path)
                print('Video {} successfully converted to wav.'.format(path))
            except IOError:
                print('ERROR, video', '\"' + path + '\"', 'does not exist.')
                sys.exit(1)

            if device == 'cuda':
                print('Tagging with CUDA, initializing GPU...')
            else:
                print('Tagging with CPU...')
            pred_array = predict_audio(audio_array, model, device, sr=16000, timestep=1,
                                       batch_size=16, n_classes=len(audio_tags))
            print('Audio tagging DONE!')
            output_path = os.path.join(args.output_path, video_name)
            # make dir for predictions
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if args.save_video:
                print('Generating video with annotations...')
                render_predict(video_name, pred_array, output_path, path,
                               tag_names=audio_tags, bitrate=args.video_bitrate,
                               gpu_encode=args.gpu_encode)
                print('Generating DONE!')
            # output predictions to json
            pred_json(video_name, pred_array, output_path, duration, tag_names=audio_tags)
            print('Predictions saved to JSON.')
            # save prediction plot
            if args.plot_predict:
                plot_predict(video_name, pred_array, output_path, tag_names=audio_tags)
                print('Predictions saved to image.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Tagging')
    parser.add_argument('--video_path', type=str, default="I:/ZKOUSKY/SIkoraP/vasulkas/videos",
                        help='path to video (or videos) for tagging')
    parser.add_argument('--model_path', type=str, default="pretrained/CNN14_Vasulka_1s.pth",
                        help='path to a pre-trained audio tagging model')
    parser.add_argument('--tag_names', type=str, default="pretrained/tag_names.txt",
                        help='tag names used by the model')
    parser.add_argument('--output_path', type=str, default="C:/_vasulka_audio",
                        help='output path for predicitions')
    parser.add_argument('--gpu_predict', type=bool, default=True,
                        help='if True, tagging runs on GPU')
    parser.add_argument('--gpu_encode', type=bool, default=True,
                        help='if True, video encoding runs on GPU')
    parser.add_argument('--save_video', type=bool, default=False,
                        help='if True, annotations are rendered to video')
    parser.add_argument('--video_bitrate', type=int, default=None,
                        help='if None, bitrate is set to video height * 4500')
    parser.add_argument('--plot_predict', type=bool, default=False,
                        help='if True, prediction plot is saved to .png')
    parser.add_argument('--skip_videos', action='store',
                        type=str, nargs='*', default=[],
                        help='skip selected videos in video_path – example: --skip_videos video1.mp4 video2.mp4')
    arguments, _ = parser.parse_known_args()
    audio_tagging(arguments)
