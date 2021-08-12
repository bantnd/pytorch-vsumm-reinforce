"""
    Generate Dataset

    1. Converting video to frames
    2. Extracting features
    3. Getting change points
    4. User Summary ( for evaluation )

"""
import sys
sys.path.append('.')
sys.path.append('../networks')
from networks.CNN import ResNet
from utils.KTS.cpd_auto import cpd_auto
from tqdm import tqdm
import math
import cv2
import numpy as np
import h5py
import argparse
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, default='data/original', help="path of video file.")
parser.add_argument('-f', '--frame', type=str, default='data/frames', help="path of frame, where will extract from videos")
parser.add_argument('--h5-gen', type=str, default='data/eccv16_dataset_tiktok2_google_pool5.h5', help="path to h5 generated file")
args = parser.parse_args()

#frame options
WIDTH=224
HIGH=224
T_FPS=5 #fps after extracted video

class Generate_Dataset:
    def __init__(self, video_path,frame_path, save_path):
        self.resnet = ResNet()
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        #self.frame_root_path = './frames'
        self.frame_root_path = Path(__file__).resolve().parent.parent/Path(frame_path)
        self.h5_file = h5py.File(save_path, 'w')

        self._set_video_list(Path(__file__).resolve().parent.parent/Path(video_path))
        print('Video path : {} H5 autogen path : {}'.format(video_path, save_path))

    def _set_video_list(self, video_path):
        if Path.is_dir(video_path):
            self.video_path = video_path
            self.video_list = sorted(Path.iterdir(video_path))
            #self.video_list.sort()
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx+1)] = {}
            self.h5_file.create_group('video_{}'.format(idx+1))


    def _extract_feature(self, frame):
        res_pool5 = self.resnet(frame)
        frame_feat = res_pool5.cpu().data.numpy().flatten()

        return frame_feat

    def _get_change_points(self, video_feat, n_frame, fps):
        print('n_frame {} fps {}'.format(n_frame, fps))
        n = n_frame / math.ceil(fps)
        m = int(math.ceil(n/2.0))
        K = np.dot(video_feat, video_feat.T)
        change_points, _ = cpd_auto(K, m, 1)
        change_points = np.concatenate(([0], change_points, [n_frame-1]))

        temp_change_points = []
        for idx in range(len(change_points)-1):
            segment = [change_points[idx], change_points[idx+1]-1]
            if idx == len(change_points)-2:
                segment = [change_points[idx], change_points[idx+1]]

            temp_change_points.append(segment)
        change_points = np.array(list(temp_change_points))

        temp_n_frame_per_seg = []
        for change_points_idx in range(len(change_points)):
            n_frame = change_points[change_points_idx][1] - change_points[change_points_idx][0]
            temp_n_frame_per_seg.append(n_frame)
        n_frame_per_seg = np.array(list(temp_n_frame_per_seg))

        return change_points, n_frame_per_seg

    # TODO : save dataset
    def _save_dataset(self):
        pass

    def generate_dataset(self):
        for video_idx, video_filename in enumerate(tqdm(self.video_list)):
        #for video_idx, video_filename in enumerate(self.video_list):
            video_path = video_filename
            #if Path.is_dir(self.video_path):
            #    video_path = os.path.join(self.video_path, video_filename)

            #video_basename = os.path.basename(video_path).split('.')[0]
            video_basename = Path(video_path).stem

            if not Path.exists(self.frame_root_path/Path(video_basename)):
                Path.mkdir(self.frame_root_path/Path(video_basename),parents=True)
            else:
                shutil.rmtree(self.frame_root_path/Path(video_basename))
                Path.mkdir(self.frame_root_path/Path(video_basename),parents=True)
            
            video_capture = cv2.VideoCapture(str(video_path))

            fps = video_capture.get(cv2.CAP_PROP_FPS)
            n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            #frame_list = []
            picks = []
            video_feat = None
            video_feat_for_train = None
            for frame_idx in tqdm(range(n_frames-1)):
                success, frame = video_capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (WIDTH, HIGH))
                if success:
                    frame_feat = self._extract_feature(frame)

                    if frame_idx % round(fps/T_FPS) == 0:
                        picks.append(frame_idx)

                        if video_feat_for_train is None:
                            video_feat_for_train = frame_feat
                        else:
                            video_feat_for_train = np.vstack((video_feat_for_train, frame_feat))
                            
                        img_filename = "{}.jpg".format(str(frame_idx).zfill(6))
                        cv2.imwrite(str(self.frame_root_path/ video_basename/ img_filename), frame)

                    if video_feat is None:
                        video_feat = frame_feat
                    else:
                        video_feat = np.vstack((video_feat, frame_feat))

                    

                else:
                    break

            video_capture.release()

            change_points, n_frame_per_seg = self._get_change_points(video_feat, n_frames, fps)

            # self.dataset['video_{}'.format(video_idx+1)]['frames'] = list(frame_list)
            # self.dataset['video_{}'.format(video_idx+1)]['features'] = list(video_feat)
            # self.dataset['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            # self.dataset['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            # self.dataset['video_{}'.format(video_idx+1)]['fps'] = fps
            # self.dataset['video_{}'.format(video_idx+1)]['change_points'] = change_points
            # self.dataset['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg

            self.h5_file['video_{}'.format(video_idx+1)]['features'] = list(video_feat_for_train)
            self.h5_file['video_{}'.format(video_idx+1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx+1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx+1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx+1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx+1)]['n_frame_per_seg'] = n_frame_per_seg

if __name__ == "__main__":
    gen = Generate_Dataset(args.path,args.frame, args.h5_gen)
    gen.generate_dataset()
    gen.h5_file.close()
