# scripts/2_extract_effects.py
import json, os, sys
import cv2
import numpy as np

def get_frame_at(video_path, second=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def color_stats(frame):
    if frame is None:
        return {}
    avg_bgr = np.mean(frame, axis=(0,1)).tolist()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv, axis=(0,1)).tolist()
    contrast = float(frame.std())
    return {"avg_bgr": avg_bgr, "avg_hsv": avg_hsv, "contrast": contrast}

if __name__ == "__main__":
    ref = sys.argv[1]
    features_in = sys.argv[2]
    out = sys.argv[3]
    with open(features_in) as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    visual_stats = []
    for (s,e) in scenes:
        mid = (s+e)/2.0 if e> s else s
        frame = get_frame_at(ref, mid)
        visual_stats.append(color_stats(frame))
    data["visual_stats"] = visual_stats
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,"w") as f:
        json.dump(data, f, indent=2)
    print("Saved visual stats to", out)
