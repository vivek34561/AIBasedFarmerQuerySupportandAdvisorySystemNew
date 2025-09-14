# 🌾 AgriAssist AI

🚀 **AI-Powered Farmer Advisory & Crop Management System**

AgriAssist AI is an end-to-end **farmer support platform** that provides **real-time query resolution, mandi price updates, crop recommendations, and disease prediction** using **LLMs, RAG pipelines, and ML models** — all accessible via **chat and voice interface**.

---

## ✨ Features

✅ **Real-Time Farmer Query Resolution** – Multilingual chatbot using **FastAPI + LangChain + Pinecone**
✅ **Mandi Price & Weather Insights** – Integrated **data.gov.in API** for daily market prices & weather-based recommendations
✅ **Personalized Crop Suggestions** – AI-driven crop recommendations based on soil, weather, and location
✅ **Voice-Enabled Interaction** – Speech-to-text (STT) & text-to-speech (TTS) for easy communication
✅ **Crop Disease Prediction** – ML-powered pipeline for early disease detection & risk reduction
✅ **Feedback & Escalation System** – Continuous improvement with **SQLite-based feedback tracking**
✅ **Interactive Dashboard** – Built with **Streamlit**, featuring chat UI and visualization

---

## 🏗️ Tech Stack

* **Backend:** FastAPI, LangChain, Pinecone, SQLite
* **Frontend:** Streamlit (Interactive Chat UI + Dashboard)
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* **Generative AI:** OpenAI GPT-4 / GPT-4o, HuggingFace Transformers
* **Speech:** OpenAI Whisper (STT), gTTS / pyttsx3 (TTS)
* **Deployment:** Docker, Render / Streamlit Cloud

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/AgriAssist-AI.git
cd AgriAssist-AI
```

### 2️⃣ Install System Dependencies

Create `apt.txt` (for Render/Streamlit Cloud):

```
portaudio19-dev
```

Or install locally (Ubuntu):

```bash
sudo apt-get update && sudo apt-get install portaudio19-dev
```

### 3️⃣ Create Virtual Environment & Install Requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4️⃣ Run Backend

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### 5️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🖼️ Project Architecture

```text
📂 AgriAssist-AI
 ┣ 📜 backend.py        # FastAPI RAG backend
 ┣ 📜 app.py            # Streamlit dashboard
 ┣ 📂 models            # ML models for disease prediction
 ┣ 📂 data              # Sample training datasets
 ┣ 📂 utils             # Helper functions
 ┣ 📜 requirements.txt  # Python dependencies
 ┣ 📜 apt.txt           # System dependencies for deployment
 ┗ 📜 README.md
```

---

## 🌐 Deployment

* **Render:** Add `apt.txt` for system deps, then deploy with
  `apt-get update && apt-get install -y portaudio19-dev && pip install -r requirements.txt`
* **Streamlit Cloud:** Just push `apt.txt` + `requirements.txt` → Auto-build

---

## 📊 Demo

🔗 **Live Demo:** [https://end-to-end-heart-disease-prediction-mhwpmnmvxn9hsahbuvxj6m.streamlit.app/](https://end-to-end-heart-disease-prediction-mhwpmnmvxn9hsahbuvxj6m.streamlit.app/)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, open issues, and submit PRs.

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.

