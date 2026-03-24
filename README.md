# Emotion-Based Music Recommender

A Streamlit app that detects your facial emotion through webcam capture and recommends songs that match your current mood.

## What Is Improved

- In-app webcam preview (no separate OpenCV popup window).
- Retry-friendly scan flow with camera warm-up.
- Confidence-based filtering for emotion predictions.
- Adjustable scan controls from sidebar:
  - scan frame count
  - warm-up frame count
  - confidence threshold
  - minimum valid predictions
  - number of recommendations
- Emotion analytics section:
  - dominant emotion
  - face-detected frame count
  - valid prediction count
  - emotion distribution chart
- Smarter recommendation mixing based on detected emotion strength.
- Download recommendations as CSV.
- Cleaner modern UI with cards, badges, and responsive layout.

## Project Files

- `app.py` - Main Streamlit application.
- `model.h5` - Trained emotion model weights.
- `dev_frontalface_default.xml` - Face detection Haar cascade.
- `muse_v3.csv` - Song metadata dataset.
- `requirements.txt` - Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser.
2. (Optional) Tune scan controls from the sidebar.
3. Click `Scan Emotion`.
4. If needed, click `Retry Scan`.
5. View emotion insights and recommended songs.
6. Export playlist using `Download Recommendations (CSV)`.

## Notes

- Good lighting and a centered face improve accuracy.
- If scan quality is weak, the app falls back to neutral recommendations.

## Author

- [Dev Dahiya](https://github.com/Dev123dahiya)
