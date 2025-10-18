# scripts/1_extract_cuts_music.py
import os
import sys
import json
import cv2
import librosa
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, threshold=30.0):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    times = [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
    video_manager.release()
    return times

def extract_audio_features(video_path, sr=22050):
    try:
        y, sr = librosa.load(video_path, sr=sr, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return {"tempo": float(tempo), "beats": [float(b) for b in librosa.frames_to_time(beats, sr=sr)]}
    except Exception as e:
        print("Audio extract error:", e)
        return {"tempo": None, "beats": []}

if __name__ == "__main__":
    ref = sys.argv[1]   # e.g. data/reference/ref1.mp4
    out = sys.argv[2]   # e.g. app/static/outputs/ref1_features.json
    scenes = detect_scenes(ref)
    audio = extract_audio_features(ref)
    result = {"ref_path": ref, "scenes": scenes, "audio": audio}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,"w") as f:
        json.dump(result, f, indent=2)
    print("Saved features to", out)
