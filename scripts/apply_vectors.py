# scripts/4_apply_vectors.py
import cv2, json, os, sys
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

def blend_frame_color(frame, avg_bgr):
    layer = np.full_like(frame, np.uint8(avg_bgr))
    return cv2.addWeighted(frame, 0.75, layer, 0.25, 0)

def process_target(target_path, features_json, output_path):
    with open(features_json) as f:
        data = json.load(f)
    visual = data.get("visual_stats", [])
    transitions = data.get("transitions", [])
    ref_audio_path = data.get("ref_path", None)

    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    num_segments = max(1, len(visual))
    seg_frames = total_frames // num_segments

    # Process segment by segment: create temp clips with moviepy for easy transitions
    temp_clips = []
    cap = cv2.VideoCapture(target_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i in range(num_segments):
        start_frame = i * seg_frames
        end_frame = (i+1)*seg_frames if i < num_segments-1 else total_frames
        temp_file = f"app/static/outputs/seg_{i}.mp4"
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width,height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for f in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret: break
            stats = visual[i] if i < len(visual) else None
            if stats and stats.get("avg_bgr"):
                frame = blend_frame_color(frame, stats["avg_bgr"])
            out.write(frame)
        out.release()
        temp_clips.append(VideoFileClip(temp_file))

    cap.release()

    # Apply transitions: if transitional type is "gradual" do crossfade
    final_clips = []
    for idx, clip in enumerate(temp_clips):
        if idx>0 and transitions and transitions[idx-1].get("type")=="gradual":
            # crossfade duration (1s)
            clip = clip.crossfadein(1)
        final_clips.append(clip)

    final = concatenate_videoclips(final_clips, method="compose")
    # overlay audio from reference if available
    if ref_audio_path:
        try:
            audio = AudioFileClip(ref_audio_path)
            final = final.set_audio(audio.subclip(0, final.duration))
        except Exception as e:
            print("Audio merge failed:", e)
    final.write_videofile(output_path, codec="libx264")
    print("Saved final to", output_path)

if __name__=="__main__":
    target = sys.argv[1]
    feats = sys.argv[2]
    out = sys.argv[3]
    process_target(target, feats, out)
