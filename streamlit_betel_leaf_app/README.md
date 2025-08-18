# 🌿 Betel Leaf Detection — Streamlit App

A minimal Streamlit app to deploy your Keras/TensorFlow betel leaf detection model.

## Folder structure
```text
.
├── app.py
├── requirements.txt
├── models/
│   └── model.h5           # put your model here (or use MODEL_URL secret)
└── .streamlit/
    ├── config.toml
    └── secrets.example.toml
```

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io and click **New app** → select your repo, branch, and `app.py`.
3. In **Advanced settings → Secrets**, add values like:
   ```toml
   # Example
   MODEL_URL   = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # or direct file link
   CLASS_NAMES = "Healthy,Diseased"
   IMG_SIZE    = "224,224"
   ```
4. Click **Deploy**. First build can take a few minutes.

### Notes
- If your `model.h5` is **<100 MB**, you can commit it to the repo under `models/`.
- If it's larger, upload it to cloud storage (e.g., Google Drive) and set `MODEL_URL` in Secrets.
- Use `CLASS_NAMES` to control the labels shown in the UI (comma-separated).
- Make sure `IMG_SIZE` matches your training input size.
