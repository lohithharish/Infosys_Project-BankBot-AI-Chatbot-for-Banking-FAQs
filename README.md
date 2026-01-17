# BankBot AI – Intelligent Chatbot for Banking FAQs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![AI](https://img.shields.io/badge/AI-NLP-green)
![LLM](https://img.shields.io/badge/LLM-Transformer--based-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Description

**BankBot AI** is an intelligent AI-powered chatbot designed to assist users with **banking-related Frequently Asked Questions (FAQs)**.
The project leverages **Natural Language Processing (NLP)** and **Large Language Models (LLMs)** to understand user queries and generate accurate, human-like responses.

This project was developed as part of an **Infosys Certification / Internship program**, focusing on real-world AI application development, modular design, and local execution.

---

## 🚀 Features

* Conversational AI chatbot for banking FAQs
* Natural language query understanding
* Context-aware responses
* LLM-powered text generation
* Modular and extensible architecture
* Works fully in a **local environment**
* Configurable LLM backend
* Clean and simple UI using Streamlit

---

## 🧠 Techniques Used

### Natural Language Processing (NLP)

* Text preprocessing
* Intent detection
* Query classification
* Tokenization and pattern matching

### Prompt Engineering

* Structured prompt templates
* Context-aware prompts
* Instruction-based response generation

### LLM-Based Text Generation

* Transformer-based language models
* Dynamic response generation
* Configurable inference settings

---

## 🛠 Tech Stack

### Programming Language

* **Python**

### Libraries / Frameworks

* Streamlit
* SQLite3
* Pandas
* Regex
* Llama-cpp
* Custom NLP inference modules

### AI / ML Technologies

* Natural Language Processing (NLP)
* Large Language Models (LLMs)
* Prompt Engineering

### LLM Details

* Uses **transformer-based LLMs**
* Supports **local LLM inference**
* **LLM is configurable** (model path, parameters, and backend can be changed without altering core logic)

---

## 📂 Project Structure

```
Infosys_Project-BankBot-AI-Chatbot-for-Banking-FAQs/
│
├── streamlit_app.py        # Main user-facing chatbot app
├── admin_app.py            # Admin dashboard
├── mainapp.py              # Application entry logic
│
├── db_files/
│   ├── db_setup.py         # Database initialization
│   ├── auth.py             # Authentication logic
│
├── nlu_engine/
│   ├── infer_intent.py     # NLP intent inference
│
├── models/
│   └── llm_model/          # Local LLM model files
│
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation Steps

1. **Clone the repository**

   ```
   git clone https://github.com/lohithharish/Infosys_Project-BankBot-AI-Chatbot-for-Banking-FAQs.git
   ```

2. **Navigate to the project directory**

   ```
   cd Infosys_Project-BankBot-AI-Chatbot-for-Banking-FAQs
   ```

3. **Create a virtual environment (recommended)**

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run the Project Locally

1. **Start the Streamlit application**

   ```
   streamlit run streamlit_app.py
   ```

2. **Access the application**

   * Open your browser
   * Go to: `http://localhost:8501`

3. **Optional**

   * Configure the LLM model path and parameters in the code if required

---

## 🎓 Certification Use Case

This project was developed as part of an **Infosys Certification / Internship Program**, demonstrating:

* Practical application of AI and NLP concepts
* LLM integration in real-world systems
* Software engineering best practices
* End-to-end project execution and documentation

The project fulfills academic and professional evaluation requirements, including **local execution, modularity, and AI integration**.

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute this project with proper attribution.

---

### 👨‍💻 Author

**MH Lohith**
B.Tech – Computer Engineering (AI & ML)
Infosys Certified Project

---

⭐ If you find this project useful, feel free to star the repository!
