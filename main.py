import pandas as pd
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# CSV laden
CSV_PATH = "/kaggle/input/chatbot-data-truly-v1/data/embedding_data_cleaned.csv"
df_embed = pd.read_csv(CSV_PATH, encoding="utf-8")
df_embed["doc_id"] = df_embed["id"] if "id" in df_embed.columns else df_embed.index

# 🔹 Dokumente erstellen
docs = []
for _, row in df_embed.iterrows():
    # Beschreibung priorisieren, ggf. Keywords einbeziehen
    combined_text = f"{row['title']}: {row['description']} {row['description']} {row.get('keywords','')}"
    docs.append(Document(
        page_content=combined_text,
        metadata={"doc_id": int(row["doc_id"]), "title": row["title"]}
    ))

# 🔹 Embeddings initialisieren
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)

# 🔹 Chroma DB laden (ohne Neuaufbau)
db_movies = Chroma(
    persist_directory="chroma_movies_bge_chatbot",
    embedding_function=embeddings
)

# 🔹 Cross-Encoder laden (Reranker)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from transformers import AutoModelForCausalLM, AutoTokenizer

# 🔹 Pfad zum Chatbot-Dataset
CHATBOT_CSV = "/kaggle/input/chatbot-data-truly-v1/data/chatbot_data_truly_cleaned.csv"
df_chatbot = pd.read_csv(CHATBOT_CSV, encoding="utf-8")

# 🔹 doc_id als Schlüssel für die Verbindung
df_chatbot["doc_id"] = df_chatbot["id"]  # Falls noch nicht vorhanden

# 🔹 Modell & Tokenizer laden (HuggingFace)
model_id = "ibm-granite/granite-3.3-2b-instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# 🔹 Funktion, um semantische Empfehlungen zu bekommen
def get_recommendations(query, top_k=5):
    recs = db_movies.similarity_search(query, k=50)
    if not recs:
        return []

    pairs = [(query, rec.page_content) for rec in recs]
    scores = reranker.predict(pairs, batch_size=8)
    
    scores_dict = {}
    for rec, score in zip(recs, scores):
        doc_id = rec.metadata["doc_id"]
        if doc_id not in scores_dict or score > scores_dict[doc_id]["score"]:
            scores_dict[doc_id] = {"score": score, "rec": rec}

    unique_top_recs = sorted(scores_dict.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [item["rec"] for item in unique_top_recs]

# 🔹 Funktion für Chatbot-Antwort
def generate_chat_response(user_message, chat_history=None, context=""):
    chat_history = chat_history or []
    
    # Verlauf zusammenbauen
    history_text = ""
    for q, a in chat_history:
        history_text += f"Frage: {q}\nAntwort: {a}\n\n"
    
    # Prompt mit Verlauf
    prompt = f"""Du bist ein lockerer Filmexperte.
Hier sind relevante Filme:
{context}

Bisheriges Gespräch:
{history_text}

Neue Frage: {user_message}
Antwort:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.9
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Antwort:")[-1].strip()
    return answer

import gradio as gr

# 🔹 Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Semantic Movie Recommender + Chatbot")

    with gr.Tab("🍿 Empfehlungen"):
        user_query = gr.Textbox(label="Beschreibe deinen idealen Film")
        submit_button = gr.Button("🔍 Finde Empfehlungen")
        output_gallery = gr.Gallery(label="Empfohlene Filme", columns=3, rows=3, show_label=True)
        hidden_text = gr.Textbox(label="Letzte Empfehlungen (unsichtbar)", visible=False)

        def submit_recommendations(query):
            recs = get_recommendations(query)
            gallery = []
            context_texts = []
            for rec in recs:
                doc_id = rec.metadata["doc_id"]
                movie_row = df_embed[df_embed["doc_id"] == doc_id].iloc[0]
                cover_url = movie_row["poster_url"] if "poster_url" in movie_row else ""
                label = f"{movie_row['title']}\n\n{movie_row['description']}"
                gallery.append((cover_url, label))
                
                # Kontext für Chatbot vorbereiten
                chatbot_row = df_chatbot[df_chatbot["doc_id"] == doc_id]
                if not chatbot_row.empty:
                    info = chatbot_row.iloc[0].to_dict()
                    context_texts.append(
                        f"{info['original_title']} ({info['release_date']}), IMDb: {info.get('vote_average','N/A')}"
                    )
            
            context_text = "\n".join(context_texts)
            return gallery, context_text

        submit_button.click(
            fn=submit_recommendations, 
            inputs=user_query, 
            outputs=[output_gallery, hidden_text]
        )

    with gr.Tab("💬 Chatbot"):
        chatbot_ui = gr.Chatbot(label="Filmexperte")
        user_msg = gr.Textbox(label="Frage den Filmexperten")
        send_btn = gr.Button("Senden")

        def chat_submit(message, chat_history, last_recommendations_text):
            chat_history = chat_history or []
            resp = generate_chat_response(
                message, chat_history=chat_history, context=last_recommendations_text
            )
            chat_history.append((message, resp))
            return chat_history, ""

        send_btn.click(
            fn=chat_submit, 
            inputs=[user_msg, chatbot_ui, hidden_text], 
            outputs=[chatbot_ui, user_msg]
        )

if __name__ == "__main__":
    demo.launch()