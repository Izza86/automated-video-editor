# scripts/3_detect_transitions.py
import json, os, sys
import cv2
import numpy as np

def frame_diff(a,b):
    return float(np.mean(cv2.absdiff(a,b)))

def detect_transition(video_path, scenes, window_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    results = []
    for (s,e) in scenes:
        start = int(max(0, s*fps))
        end = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1, e*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        prev = None
        diffs = []
        frames_to_check = min(end-start, int(window_seconds*fps))
        for i in range(frames_to_check):
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                diffs.append(frame_diff(prev, gray))
            prev = gray
        avg = np.mean(diffs) if diffs else 0.0
        ttype = "cut" if avg > 30 else ("gradual" if avg > 5 else "static")
        results.append({"scene":(s,e), "avg_diff": avg, "type": ttype})
    cap.release()
    return results

if __name__=="__main__":
    video = sys.argv[1]
    feats_in = sys.argv[2]
    out = sys.argv[3]
    with open(feats_in) as f:
        data = json.load(f)
    scenes = data.get("scenes", [])
    trans = detect_transition(video, scenes)
    data["transitions"] = trans
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,"w") as f:
        json.dump(data, f, indent=2)
    print("Saved transitions to", out)
