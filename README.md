# 🧠 HealthAI: Intelligent Healthcare Assistant Using IBM Granite
HealthAI is a smart healthcare assistant powered by IBM Watson Machine Learning and the Granite 13B model. It provides real-time, AI-driven support for users seeking accurate health insights, symptom prediction, home remedies, treatment suggestions, and vital analytics.

# 🎯 Project Overview
HealthAI helps users:

Understand their symptoms
Predict possible health conditions
Receive natural remedies
Get AI-generated treatment plans
Monitor health metrics (heart rate, BP, glucose)
Receive daily health tips
The system uses FastAPI, HTML/CSS/JavaScript SPA frontend, and IBM WatsonX Granite 13B Instruct v2.

# 🧩 Features
## 💬 Patient Chat – AI-powered health Q&A with empathetic responses
🧪 Disease Identifier – Predicts conditions based on user symptoms
🌿 Home Remedies – Natural solutions for common conditions
💊 Treatment Plan Generator – Personalized plans with medications and lifestyle tips
📊 Health Dashboard – Visualizes vitals and provides AI insights using Chart.js
💡 Health Tips – 6 personalized wellness tips per day in an attractive UI
# 🏗️ Architecture
<h3>Frontend</h3>: HTML, CSS, JavaScript (Single Page Application)
<h3>Backend</h3>: FastAPI (Python)
AI Engine: IBM WatsonX Granite 13B v2
Visualization: Chart.js
# 🔄 Layers
UI Layer – Chat interface, health data forms, visualizations
Application Layer – FastAPI handles routing, sessions, and IBM model integration
Data Layer – Session and sample patient data
AI Layer – IBM Granite responds to symptom inputs and treatment prompts
📁 Folder Structure
HealthAI/
├── static/                      # All frontend assets
│   ├── css/
│   │   └── style.css            # UI styles for entire SPA
│   ├── img/
│   │   └── bg.jpg               # Background image
│   └── js/
│       └── app.js              # JavaScript for routing and API calls
│
├── templates/
│   └── index.html              # Main frontend layout rendered by FastAPI
│
├── main.py                     # FastAPI backend logic + routes
├── .env                        # IBM Watson API keys and config (keep secret)
├── venv/                       # Python virtual environment
└── __pycache__/                # Python bytecode cache (auto-generated)


## ⚙️ Setup Instructions

### ✅ Prerequisites

- Python 3.8+
- Git
- IBM Cloud account with Watson Machine Learning access

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/HealthAI.git
cd HealthAI

# Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
📌 Create a .env file in the root directory with your IBM WatsonX credentials:

IBM_API_KEY="your-ibm-api-key"
IBM_GRANITE_ENDPOINT="your-model-endpoint"
IBM_MODEL_ID="granite-model-id"
IBM_PROJECT_ID="your-ibm-project-id"
▶️ Run the App
uvicorn main:app --reload
Visit: http://127.0.0.1:8000

👥 Contributors
Nikku Chandana Sai Durga
Mudili Lakshmi Srimannarayana Charan Teja
Muppalla Sindhu
Motukuri Shrivalli
Feel free to contribute or fork this project for expansion!

📄 License
This project is licensed under the MIT License.
