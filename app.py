from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "muse_v3.csv"
MODEL_PATH = BASE_DIR / "model.h5"
LOCAL_CASCADE_PATH = BASE_DIR / "dev_frontalface_default.xml"

EMOTION_DICT = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised",
}

# Map model emotions to recommendation buckets.
EMOTION_BUCKET_MAP = {
    "Angry": "angry",
    "Disgusted": "angry",
    "Fearful": "fear",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
    "Surprised": "happy",
}


@st.cache_data(show_spinner=False)
def load_music_data():
    data = pd.read_csv(CSV_PATH)

    data["link"] = data["lastfm_url"]
    data["name"] = data["track"]
    data["emotional"] = data["number_of_emotion_tags"]
    data["pleasant"] = data["valence_tags"]

    data = data[["name", "emotional", "pleasant", "link", "artist"]]
    data = data.dropna(subset=["name", "artist", "link"]).reset_index(drop=True)
    data = data.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

    # Dynamic split so recommendation slices stay stable even if dataset size changes.
    slices = np.array_split(data, 5)
    return {
        "sad": slices[0].reset_index(drop=True),
        "fear": slices[1].reset_index(drop=True),
        "angry": slices[2].reset_index(drop=True),
        "neutral": slices[3].reset_index(drop=True),
        "happy": slices[4].reset_index(drop=True),
    }


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    model.load_weights(str(MODEL_PATH))
    return model


@st.cache_resource(show_spinner=False)
def load_cascade_classifier():
    if LOCAL_CASCADE_PATH.exists():
        classifier = cv2.CascadeClassifier(str(LOCAL_CASCADE_PATH))
        if not classifier.empty():
            return classifier

    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def inject_ui_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #0f172a;
            --bg-card: rgba(15, 23, 42, 0.65);
            --line: rgba(148, 163, 184, 0.3);
            --text-main: #e2e8f0;
            --text-muted: #94a3b8;
            --accent: #14b8a6;
            --accent-soft: #0ea5a5;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
        }

        .stApp {
            background:
                radial-gradient(1000px 500px at 5% -10%, rgba(20, 184, 166, 0.25), transparent 55%),
                radial-gradient(900px 400px at 100% 0%, rgba(249, 115, 22, 0.22), transparent 50%),
                linear-gradient(145deg, #020617 0%, #0f172a 40%, #111827 100%);
            color: var(--text-main);
        }

        .hero {
            border: 1px solid var(--line);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.75));
            border-radius: 20px;
            padding: 1.3rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 20px 40px rgba(2, 6, 23, 0.4);
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(1.5rem, 2.5vw, 2.2rem);
            font-weight: 700;
            letter-spacing: 0.2px;
            color: #f8fafc;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: var(--text-muted);
            font-size: 0.98rem;
        }

        .panel {
            border: 1px solid var(--line);
            background: var(--bg-card);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-top: 0.9rem;
            backdrop-filter: blur(3px);
        }

        .panel-title {
            margin: 0 0 0.6rem;
            font-size: 1.02rem;
            font-weight: 600;
            color: #f1f5f9;
        }

        .emotion-badge {
            display: inline-block;
            margin: 0.2rem 0.35rem 0.2rem 0;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
            border: 1px solid rgba(20, 184, 166, 0.35);
            background: rgba(20, 184, 166, 0.14);
            color: #99f6e4;
            font-weight: 600;
            font-size: 0.87rem;
        }

        .song-card {
            border: 1px solid var(--line);
            background: rgba(15, 23, 42, 0.72);
            border-radius: 14px;
            padding: 0.82rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .song-card a {
            color: #f8fafc !important;
            text-decoration: none;
            font-weight: 600;
        }

        .song-card a:hover {
            color: #5eead4 !important;
        }

        .song-artist {
            color: var(--text-muted);
            margin-top: 0.3rem;
            font-size: 0.9rem;
        }

        .stButton > button {
            border-radius: 12px;
            padding: 0.5rem 1.05rem;
            border: 1px solid var(--line);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, var(--accent), var(--accent-soft));
            color: white;
            border: none;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def detect_faces_multi(gray_frame, classifier):
    candidates = []
    detection_passes = [
        (gray_frame, 1.25, 5, (30, 30)),
        (gray_frame, 1.15, 4, (24, 24)),
        (cv2.equalizeHist(gray_frame), 1.2, 4, (24, 24)),
    ]

    for img, scale_factor, min_neighbors, min_size in detection_passes:
        faces = classifier.detectMultiScale(
            img,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(faces) > 0:
            candidates = faces
            break

    return candidates


def summarize_emotions(emotions):
    counts = Counter(emotions)
    ranked = [item for item, _ in counts.most_common()]
    return ranked, counts


def build_sampling_plan(emotion_counts, total_songs):
    if not emotion_counts:
        return {"neutral": total_songs}

    bucket_counts = Counter()
    for emotion_name, count in emotion_counts.items():
        bucket = EMOTION_BUCKET_MAP.get(emotion_name, "neutral")
        bucket_counts[bucket] += count

    total_votes = sum(bucket_counts.values())
    raw_allocations = {
        bucket: (count / total_votes) * total_songs
        for bucket, count in bucket_counts.items()
    }

    final_allocations = {bucket: int(value) for bucket, value in raw_allocations.items()}
    assigned = sum(final_allocations.values())

    # Distribute remaining songs based on largest decimal remainders.
    remainder_order = sorted(
        raw_allocations.items(),
        key=lambda item: item[1] - int(item[1]),
        reverse=True,
    )
    idx = 0
    while assigned < total_songs and remainder_order:
        bucket = remainder_order[idx % len(remainder_order)][0]
        final_allocations[bucket] += 1
        assigned += 1
        idx += 1

    # Ensure at least one recommendation from dominant mood.
    dominant_bucket = bucket_counts.most_common(1)[0][0]
    if final_allocations.get(dominant_bucket, 0) == 0:
        final_allocations[dominant_bucket] = 1

    # Rebalance if we overshoot due to guard rails.
    while sum(final_allocations.values()) > total_songs:
        removable = [b for b, c in final_allocations.items() if c > 0 and b != dominant_bucket]
        if not removable:
            break
        final_allocations[removable[0]] -= 1

    return final_allocations


def recommend_songs(emotion_counts, data_slices, total_songs=30):
    sampling_plan = build_sampling_plan(emotion_counts, total_songs)
    chunks = []

    for bucket, take_count in sampling_plan.items():
        if take_count <= 0:
            continue

        source = data_slices.get(bucket, data_slices["neutral"])
        replace = take_count > len(source)
        chunks.append(source.sample(n=take_count, replace=replace))

    if chunks:
        recommendations = pd.concat(chunks, ignore_index=True)
    else:
        recommendations = data_slices["neutral"].sample(n=total_songs).reset_index(drop=True)

    recommendations = recommendations.drop_duplicates(subset=["name", "artist"])

    if len(recommendations) < total_songs:
        needed = total_songs - len(recommendations)
        refill = data_slices["neutral"].sample(n=needed, replace=needed > len(data_slices["neutral"]))
        recommendations = pd.concat([recommendations, refill], ignore_index=True)
        recommendations = recommendations.drop_duplicates(subset=["name", "artist"])

    return recommendations.head(total_songs).reset_index(drop=True)


def scan_emotions(
    model,
    face_classifier,
    total_frames=50,
    warmup_frames=12,
    confidence_threshold=0.55,
    min_valid_predictions=6,
):
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        return None, "Camera is not available. Please close other apps using the webcam.", "error"

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    status_placeholder.info("Warming up camera...")

    for _ in range(warmup_frames):
        ok, _ = capture.read()
        if not ok:
            break
        time.sleep(0.02)

    status_placeholder.info("Scanning emotion... keep your face centered.")
    accepted_predictions = []
    face_detected_frames = 0

    frame_count = 0
    while frame_count < total_frames:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces_multi(gray, face_classifier)
        frame_count += 1

        if len(faces) > 0:
            face_detected_frames += 1
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            cv2.rectangle(frame, (x, y - 45), (x + w, y + h + 10), (20, 184, 166), 2)

            roi_gray = gray[y : y + h, x : x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img, verbose=0)[0]
            max_index = int(np.argmax(prediction))
            detected_emotion = EMOTION_DICT[max_index]
            confidence = float(np.max(prediction))

            if confidence >= confidence_threshold:
                accepted_predictions.append(detected_emotion)

            cv2.putText(
                frame,
                f"{detected_emotion} ({confidence:.2f})",
                (x + 8, y - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        frame_placeholder.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            caption=f"Scanning frame {frame_count}/{total_frames}",
            use_container_width=True,
        )

    capture.release()
    frame_placeholder.empty()
    status_placeholder.empty()

    if len(accepted_predictions) < min_valid_predictions:
        fallback_counts = Counter({"Neutral": 1})
        result = {
            "ranked_emotions": ["Neutral"],
            "emotion_counts": fallback_counts,
            "accepted_predictions": len(accepted_predictions),
            "face_detected_frames": face_detected_frames,
            "total_frames": frame_count,
        }
        return (
            result,
            "Face signal was weak in this scan, so neutral recommendations are shown. Try better lighting or move closer to camera.",
            "info",
        )

    ranked_emotions, emotion_counts = summarize_emotions(accepted_predictions)
    result = {
        "ranked_emotions": ranked_emotions,
        "emotion_counts": emotion_counts,
        "accepted_predictions": len(accepted_predictions),
        "face_detected_frames": face_detected_frames,
        "total_frames": frame_count,
    }

    return result, "Emotions detected successfully.", "success"


def init_state():
    if "scan_result" not in st.session_state:
        st.session_state.scan_result = None
    if "scan_message" not in st.session_state:
        st.session_state.scan_message = ""
    if "scan_message_type" not in st.session_state:
        st.session_state.scan_message_type = "info"
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []


def build_history_entry(scan_result):
    counts = dict(scan_result.get("emotion_counts", {}))
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ranked_emotions": list(scan_result.get("ranked_emotions", [])),
        "emotion_counts": counts,
        "accepted_predictions": int(scan_result.get("accepted_predictions", 0)),
        "face_detected_frames": int(scan_result.get("face_detected_frames", 0)),
        "total_frames": int(scan_result.get("total_frames", 0)),
    }


def add_scan_to_history(scan_result, max_items=5):
    entry = build_history_entry(scan_result)
    st.session_state.scan_history.insert(0, entry)
    st.session_state.scan_history = st.session_state.scan_history[:max_items]


def history_entry_to_result(entry):
    return {
        "ranked_emotions": list(entry.get("ranked_emotions", [])),
        "emotion_counts": Counter(entry.get("emotion_counts", {})),
        "accepted_predictions": int(entry.get("accepted_predictions", 0)),
        "face_detected_frames": int(entry.get("face_detected_frames", 0)),
        "total_frames": int(entry.get("total_frames", 0)),
    }


def show_message(msg, msg_type):
    if not msg:
        return
    if msg_type == "success":
        st.success(msg)
    elif msg_type == "error":
        st.error(msg)
    else:
        st.info(msg)


def render_song_list(recommended_df):
    st.markdown("<div class='panel'><p class='panel-title'>Recommended Songs</p>", unsafe_allow_html=True)
    for idx, row in recommended_df.iterrows():
        song_html = (
            f"<div class='song-card'>"
            f"<a href='{row['link']}' target='_blank'>{idx + 1}. {row['name']}</a>"
            f"<div class='song-artist'>{row['artist']}</div>"
            f"</div>"
        )
        st.markdown(song_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_detection_insights(scan_result):
    if not scan_result:
        return

    counts = scan_result["emotion_counts"]
    if not counts:
        return

    dominant_emotion = counts.most_common(1)[0][0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Dominant Emotion", dominant_emotion)
    m2.metric("Face Detected Frames", f"{scan_result['face_detected_frames']}/{scan_result['total_frames']}")
    m3.metric("Valid Predictions", scan_result["accepted_predictions"])

    chart_df = pd.DataFrame(
        {"Emotion": list(counts.keys()), "Count": list(counts.values())}
    ).sort_values(by="Count", ascending=False)
    st.bar_chart(chart_df, x="Emotion", y="Count", use_container_width=True)


def download_recommendations_button(recommendations):
    csv_data = recommendations[["name", "artist", "link"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Recommendations (CSV)",
        data=csv_data,
        file_name="emotion_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_history_panel():
    st.markdown("<div class='panel'><p class='panel-title'>Recent Scans</p>", unsafe_allow_html=True)
    history = st.session_state.scan_history

    if not history:
        st.caption("No previous scans yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for idx, entry in enumerate(history):
        dominant = entry["ranked_emotions"][0] if entry["ranked_emotions"] else "Neutral"
        left, load_col, del_col = st.columns([3, 1, 1])
        left.markdown(
            f"**{entry['timestamp']}**  \n"
            f"Dominant: `{dominant}` | Valid: `{entry['accepted_predictions']}`"
        )
        if load_col.button("Load", key=f"load_hist_{idx}", use_container_width=True):
            st.session_state.scan_result = history_entry_to_result(entry)
            st.session_state.scan_message = f"Loaded scan from {entry['timestamp']}."
            st.session_state.scan_message_type = "info"
            st.rerun()
        if del_col.button("Delete", key=f"del_hist_{idx}", use_container_width=True):
            st.session_state.scan_history.pop(idx)
            st.rerun()

    if st.button("Clear History", key="clear_all_history", use_container_width=True):
        st.session_state.scan_history = []
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Emotion Music Recommender", page_icon="M", layout="wide")
    inject_ui_styles()
    init_state()

    music_data = load_music_data()
    model = load_emotion_model()
    face_classifier = load_cascade_classifier()

    with st.sidebar:
        st.header("Scan Controls")
        total_frames = st.slider("Scan Frames", min_value=30, max_value=90, value=55, step=5)
        warmup_frames = st.slider("Warm-up Frames", min_value=0, max_value=30, value=14, step=1)
        confidence_threshold = st.slider("Confidence Threshold", min_value=0.30, max_value=0.90, value=0.55, step=0.05)
        min_valid_predictions = st.slider("Min Valid Predictions", min_value=3, max_value=20, value=6, step=1)
        total_recommendations = st.slider("Total Recommendations", min_value=10, max_value=50, value=30, step=5)
        st.caption("Higher threshold means stricter emotion acceptance.")

    st.markdown(
        """
        <div class="hero">
            <h1>Emotion-Based Music Recommender</h1>
            <p>Scan your face, detect your mood with confidence filters, and get a curated playlist instantly.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 1.5, 1])

    with center_col:
        st.markdown("<div class='panel'><p class='panel-title'>Emotion Scan</p>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        start_scan = c1.button("Scan Emotion", type="primary", use_container_width=True)
        retry_scan = c2.button("Retry Scan", use_container_width=True)
        clear_scan = c3.button("Clear", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if clear_scan:
            st.session_state.scan_result = None
            st.session_state.scan_message = "Scan cleared."
            st.session_state.scan_message_type = "info"

        if face_classifier.empty():
            st.error("Face detector could not be loaded. Please verify the cascade XML file.")
        elif start_scan or retry_scan:
            result, message, msg_type = scan_emotions(
                model,
                face_classifier,
                total_frames=total_frames,
                warmup_frames=warmup_frames,
                confidence_threshold=confidence_threshold,
                min_valid_predictions=min_valid_predictions,
            )
            st.session_state.scan_result = result
            st.session_state.scan_message = message
            st.session_state.scan_message_type = msg_type
            if result is not None:
                add_scan_to_history(result)

        show_message(st.session_state.scan_message, st.session_state.scan_message_type)

        if st.session_state.scan_result:
            badges = "".join(
                [
                    f"<span class='emotion-badge'>{emotion}</span>"
                    for emotion in st.session_state.scan_result["ranked_emotions"]
                ]
            )
            st.markdown(
                f"<div class='panel'><p class='panel-title'>Detected Emotions</p>{badges}</div>",
                unsafe_allow_html=True,
            )

    scan_result = st.session_state.scan_result
    emotion_counts = scan_result["emotion_counts"] if scan_result else Counter({"Neutral": 1})

    recommendations = recommend_songs(
        emotion_counts=emotion_counts,
        data_slices=music_data,
        total_songs=total_recommendations,
    )

    render_detection_insights(scan_result)
    render_history_panel()
    download_recommendations_button(recommendations)
    render_song_list(recommendations)


if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(False)
    main()
