# DermAI Pro — Skin Disease Prediction System

> **100% offline · No API key · No cloud · Runs locally on your machine**

A professional AI-powered dermatology assistant that classifies **23 skin disease categories** using a custom-trained deep learning model. Built with Python Flask backend + TensorFlow/Keras + a stunning dark UI.

---

## 🏗️ Project Structure

```
dermai/
├── app.py              ← Flask backend (main server)
├── train_model.py      ← Model training script
├── start.sh            ← One-click startup
├── model/
│   ├── skin_model.h5   ← Trained CNN model (auto-created)
│   ├── classes.json    ← 23 disease class names
│   └── disease_info.json ← Full medical knowledge base
├── templates/
│   └── index.html      ← Beautiful frontend UI
└── static/             ← Static assets (if any)
```

---

## 🚀 Quick Start

### Option A — One command
```bash
chmod +x start.sh
./start.sh
```

### Option B — Manual
```bash
# 1. Install dependencies
pip install flask tensorflow pillow numpy flask-cors

# 2. Train the model (only needed once)
python3 train_model.py

# 3. Start the server
python3 app.py

# 4. Open browser
open http://localhost:5000
```

---

## 🧬 23 Skin Disease Classes

| # | Disease | Urgency |
|---|---------|---------|
| 1 | Acne & Rosacea | Routine |
| 2 | Actinic Keratosis & Malignant Lesions | Soon |
| 3 | Atopic Dermatitis (Eczema) | Routine |
| 4 | Bullous Disease | **Urgent** |
| 5 | Cellulitis & Bacterial Infections | **Urgent** |
| 6 | Eczema | Routine |
| 7 | Exanthems & Drug Eruptions | Soon |
| 8 | Hair Loss & Alopecia | Routine |
| 9 | Herpes HPV & STDs | Soon |
| 10 | Light Diseases & Pigmentation | Routine |
| 11 | Lupus & Connective Tissue | Soon |
| 12 | **Melanoma Skin Cancer** | **🚨 Urgent** |
| 13 | Nail Fungus & Nail Disease | Routine |
| 14 | Poison Ivy & Contact Dermatitis | Routine |
| 15 | Psoriasis & Lichen Planus | Routine |
| 16 | Scabies Lyme & Parasitic Infections | Soon |
| 17 | Seborrheic Keratosis | Routine |
| 18 | Skin Tags | Routine |
| 19 | Tinea Ringworm & Fungal Infections | Routine |
| 20 | Urticaria (Hives) | Routine |
| 21 | Vascular Tumors | Routine |
| 22 | Vasculitis | Soon |
| 23 | Warts & Molluscum | Routine |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.x + Flask |
| **AI Model** | TensorFlow 2.x / Keras CNN |
| **Image Processing** | Pillow (PIL) + NumPy |
| **Frontend** | HTML5 + CSS3 + Vanilla JavaScript |
| **Fonts** | Syne, JetBrains Mono, Lora (Google Fonts) |

**No API keys. No internet required after first load.**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve the frontend |
| POST | `/api/predict` | Run prediction on image |
| GET | `/api/classes` | List all 23 disease classes |
| GET | `/api/health` | Check server + model status |

### POST `/api/predict` — Request body (JSON)
```json
{
  "image_b64": "<base64-encoded image>",
  "age": "34",
  "gender": "Female",
  "skin_type": "Type III – Medium",
  "duration": "1–4 weeks",
  "area": "left forearm",
  "symptoms": ["Itching", "Redness", "Scaling"]
}
```

---

## 🏋️ Improving Model Accuracy

The included model is trained on **synthetic data** for demonstration. To get real clinical accuracy:

1. Download the **HAM10000** dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
2. Or use the **ISIC Archive** at https://www.isic-archive.com
3. Update `train_model.py` to use real images organised in class folders
4. Run `python3 train_model.py` — the model will retrain on real data

---

## ⚕️ Medical Disclaimer

This system is intended for **educational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified dermatologist for proper diagnosis and treatment.

---

## 📄 License

MIT — Free to use, modify, and distribute.
