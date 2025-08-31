
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import cv2

from detectors import text_emotion, face_emotion

st.set_page_config(page_title="Emotion Detector", page_icon=":mag:", layout="wide")
st.title("Emotion Detector (Text + Face)")

with st.expander("About this demo"):
    st.write(
        "This app classifies emotions from text using a distilled GoEmotions model "
        "and estimates facial emotions on uploaded images using the FER library. "
        "It's meant for learning and prototyping — not for making decisions about people."
    )

tab_text, tab_face = st.tabs(["📝 Text Emotions", "🖼️ Face Emotions"])

with tab_text:
    st.subheader("Text Emotion Analysis")
    text = st.text_area("Enter text", height=150, placeholder="Type a sentence or paragraph...")
    top_k = st.slider("Top K labels", min_value=1, max_value=6, value=5)
    if st.button("Analyze Text", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                preds = text_emotion.predict(text, top_k=top_k)
            if not preds:
                st.info("No prediction available.")
            else:
                df = pd.DataFrame(preds)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("label")["score"])
                st.success(f"Top emotion: **{df.iloc[0]['label']}** ({df.iloc[0]['score']:.2f})")

with tab_face:
    st.subheader("Facial Emotion from Image")
    uploaded = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg","jpeg","png"])
    detect = st.button("Detect Emotions", type="primary", disabled=uploaded is None)
    if uploaded is None:
        st.caption("Upload a clear, front-facing image with good lighting.")
    if uploaded and detect:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)
        with st.spinner("Detecting faces and emotions..."):
            results = face_emotion.detect(image_np)
        annotated = image_np.copy()
        for r in results:
            x, y, w, h = r["box"]
            emo = r["top_emotion"]
            conf = r["confidence"]
            # green box + label
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{emo} {conf:.2f}",
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        st.image(annotated, caption="Detections", use_column_width=True)
        if results:
            st.write("Per-face emotion scores:")
            for idx, r in enumerate(results, 1):
                st.write(f"**Face {idx}**")
                emo_scores = pd.DataFrame([r["emotions"]]).T
                emo_scores.columns = ["score"]
                st.bar_chart(emo_scores["score"])
        else:
            st.info("No faces detected. Try a clearer, front-facing image.")
