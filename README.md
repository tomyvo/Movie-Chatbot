# 🤖🎬 Movie Chatbot

![Repo Status](https://img.shields.io/badge/Status-Ready-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-555555?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **“Chat your way to your next favorite movie.”**  
> A semantic movie chatbot that leverages **three LLMs**: one for embeddings, one for precise reranking, and a third for generating natural, conversational responses.

---

## 🚀 What It Does

- 🧠 **Semantic search** over a movie dataset using embeddings + reranking  
- 💬 **Chatbot interface** powered by an LLM for natural, conversational recommendations  
- 🎨 Interactive **Gradio chat UI** for seamless user experience  
- 📚 Returns recommended movies with posters, descriptions, and reasoning  

---

## ✨ Features

- 🗨️ **Full chatbot interface** – ask questions, get movie suggestions naturally  
- 🔍 **Two-stage retrieval** → Embedding search + Cross-Encoder reranking  
- 🎯 **Chat LLM** generates conversational answers, explains recommendations  
- 🖼️ **Poster previews** for each suggested movie  
- 🌈 **Gradio Blocks UI** with clean and modern theme  
- 🛠️ **Easily extendable** – swap models, add datasets, or customize responses  

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)  
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black)  
![SentenceTransformers](https://img.shields.io/badge/SBERT-2C2C2C?style=for-the-badge&logo=pytorch&logoColor=white)  
![Gradio](https://img.shields.io/badge/Gradio-555555?style=for-the-badge&logo=gradio&logoColor=white)  

---

## 📸 Demo – Example Queries

Try the following queries in the Semantic Movie Recommender:

- **"A movie about fast cars"**  
- **"A movie about magic"**  
- **"A movie about sex and romance"**

Output examples (movie posters + descriptions):

<p align="center">
  <img src="./images/fast_cars.png" alt="Fast Cars" width="30%" />
  <img src="./images/magic.png" alt="Magic" width="30%" />
  <img src="./images/romance.png" alt="Sex & Romance" width="30%" />
</p>

---


## ⚡ Quickstart

I would recommend you to use my kaggle notebook to start and demo the semantic recommender
(https://www.kaggle.com/code/tomy00/netflix-recommender)

Or otherwise you use the code yourself


```bash
# Clone repo
git clone https://github.com/YourUser/Semantic-Movie-Chatbot.git
cd Semantic-Movie-Chatbot

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the chatbot
python main.py
