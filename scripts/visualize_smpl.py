"""Export an offline SMPL point-cloud viewer aligned to original video frames."""
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from smpl_data import load_numeric

ROOT = Path(__file__).resolve().parents[1]
C = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def export(source, output, robot_dir=None):
    output.mkdir(parents=True, exist_ok=True)
    clips = []
    for index, path in enumerate(sorted(source.glob('*.pkl'))):
        d = load_numeric(path)['merged']
        video = path.with_suffix('.mp4')
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        ids = d['frame_ids'].astype(int)
        if fps <= 0 or np.any(np.diff(ids) <= 0) or ids[-1] >= video_frames:
            raise ValueError(f'Invalid frame alignment: {path}')
        vertices = d['verts'] @ C.T
        if not np.isfinite(vertices).all():
            raise ValueError(f'Nonfinite vertices: {path}')
        vertices -= np.array([*vertices[0, :, :2].mean(0), 0])
        payload = dict(name=path.stem, fps=fps, ids=ids.tolist(),
                       points=np.round(vertices[:, ::6], 3).tolist(),
                       video=os.path.relpath(video, output), robot=None)
        robot_path = robot_dir / (path.stem + '.preview.npz') if robot_dir else None
        if robot_path and robot_path.exists():
            r = np.load(robot_path, allow_pickle=False)
            points = r['points'].copy()
            points[:, :, :2] -= points[0, 0, :2]
            payload['robot'] = np.round(points, 4).tolist()
            payload['edges'] = r['edges'].tolist()
        filename = f'clip_{index:02d}.js'
        (output / filename).write_text('window.setClip(' + json.dumps(payload, separators=(',', ':')) + ');')
        clips.append(dict(name=path.stem, file=filename, frames=len(ids), fps=fps,
                          video_start_frame=int(ids[0])))
    template = (ROOT / 'scripts/smpl_viewer.html').read_text()
    (output / 'index.html').write_text(template.replace('__CLIPS__', json.dumps(clips, ensure_ascii=False)))
    (output / 'inventory.json').write_text(json.dumps(clips, ensure_ascii=False, indent=2))
    print(f'Exported {len(clips)} clips: {output / "index.html"}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT / 'data/ForehandClear_video_SMPL_pair')
    p.add_argument('--output', type=Path, default=ROOT / 'data/ForehandClear_visualization')
    p.add_argument('--robot-dir', type=Path)
    a = p.parse_args()
    export(a.input, a.output, a.robot_dir)
