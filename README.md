# 🌾 AgroAssist AI – AI-Powered Farmer Advisory System  

AgroAssist AI is an **AI-powered farmer advisory platform** that helps farmers get **real-time answers** to agricultural queries, **view mandi prices**, receive **weather-based crop recommendations**, and even **predict crop diseases** — all in their **native language** with voice support.

## 🚀 Features
- **🧠 AI-Powered Advisory** – Built using **FastAPI, LangChain, Pinecone**, enabling **real-time, multilingual query resolution**.
- **📈 Mandi Prices & Weather** – Fetches **live mandi prices** and gives **weather-based crop suggestions** for better decision-making.
- **🎙️ Voice Support** – Integrated **speech-to-text** for hands-free usage.
- **📊 Interactive Dashboard** – Streamlit-based UI with **chat interface, voice input**, and **personalized recommendations**.
- **📢 Feedback & Escalation System** – SQLite-powered feedback collection for **continuous improvement**.
- **🩺 Disease Prediction** – ML-powered pipeline for **early detection**, helping farmers reduce crop losses.

---

## 🖼️ Screenshots  

### 🏠 Dashboard – Home Page  
![Dashboard Screenshot](./Screenshot%202025-09-14%20101417.png)

### 🤖 AI Chatbot – Real-time Query Resolution  
![Chatbot Screenshot](./Screenshot%202025-09-14%20104347.png)

### 📈 Mandi Price & Crop Recommendation  
![Mandi Price Screenshot](./Screenshot%202025-09-14%20104405.png)

### 🩺 Disease Prediction  
![Disease Prediction Screenshot](./Screenshot%202025-09-14%20104448.png)

### 📢 Feedback & Escalation System  
![Feedback Screenshot](./Screenshot%202025-09-14%20104508.png)

---

## 🛠️ Tech Stack  

- **Backend:** FastAPI, LangChain, Pinecone, SQLite  
- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn, XGBoost, CatBoost  
- **LLM & Embeddings:** OpenAI GPT-4/Groq, HuggingFace Transformers  
- **APIs Integrated:** Data.gov.in Mandi Price API, Weather API  
- **Voice Support:** OpenAI Whisper (STT), gTTS (TTS)  

---

## ⚙️ Installation & Setup  

```bash
# 1️⃣ Clone the Repository
git clone https://github.com/yourusername/agroassist-ai.git
cd agroassist-ai

# 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3️⃣ Install Dependencies
pip install -r requirements.txt

# 4️⃣ Setup Environment Variables
cp .env.example .env
# Fill in API keys (OpenAI, Pinecone, Weather API, etc.) in .env file

# 5️⃣ Run Backend (FastAPI)
uvicorn backend:app --reload

# 6️⃣ Run Frontend (Streamlit)
streamlit run app.py
