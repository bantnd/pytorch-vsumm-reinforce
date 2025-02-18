import h5py
import cv2
from pathlib import Path
from config import config


def frm2video(frm_dir, summary, vid_writer):

    for idx, val in enumerate(summary):
        if val == 1:
            # here frame name starts with '000000.jpg'
            # change according to your need
            frm_name = str(idx).zfill(6) + '.jpg'
            frm_path = Path(__file__).resolve().parent/Path(frm_dir)/ Path(frm_name)
            print(frm_path)
            frm = cv2.imread(str(frm_path))
            frm = cv2.resize(frm, (config.Width, config.Height))
            vid_writer.write(frm)

if __name__ == '__main__':
    if not Path.exists(Path(__file__).resolve().parent/Path(config.Out_dir)):
        Path.mkdir(Path(__file__).resolve().parent/Path(config.Out_dir),parents=True)

    vid_writer = cv2.VideoWriter(
        str(Path(__file__).resolve().parent/Path(config.Out_dir)/Path(config.Save_name)),
        cv2.VideoWriter_fourcc(*'MP4V'),
        config.Fps,
        (config.Width, config.Height),
    )
    h5_res = h5py.File(str(Path(__file__).resolve().parent/ Path(config.Path)/ 'result.h5'), 'r')
    key = list(h5_res.keys())[config.Idx]
    summary = h5_res[key]['machine_summary'][...]
    h5_res.close()
    frm2video(Path(__file__).resolve().parent/Path(config.Frm_dir)/Path(key), summary, vid_writer)
    vid_writer.release()