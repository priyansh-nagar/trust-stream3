import streamlit as st, uuid, os, shutil
from PIL import Image
from model.predict import predict
from model.heatmap import cam_heatmap
from feedback.db import init_db, save

init_db()
st.set_page_config("DeepTrust", "🛡️")
st.title("🛡️ DeepTrust – AI Image Deepfake Detector")
st.caption("Building trust in digital media using explainable AI")

file = st.file_uploader("Upload an image", ["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    label, conf, trust, tensor, model = predict(img)

    st.metric("Trust Score", f"{trust}/100")
    st.progress(trust/100)

    result = "AI-Generated ❌" if label else "Real Image ✅"
    st.subheader(result)

    os.makedirs("data/uploads", exist_ok=True)
    path = f"data/uploads/{uuid.uuid4()}.jpg"
    img.save(path)

    if st.checkbox("Show explanation"):
        heat = cam_heatmap(model, tensor, img)
        st.image(heat, caption="Model Attention Areas")

    col1, col2 = st.columns(2)
    if col1.button("Correct"):
        save(path, label, label)
        st.success("Feedback saved")

    if col2.button("Wrong"):
        correct = 0 if label else 1
        save(path, label, correct)
        target = "fake" if correct else "real"
        os.makedirs(f"data/labeled/{target}", exist_ok=True)
        shutil.copy(path, f"data/labeled/{target}/")
        st.warning("Saved for retraining")