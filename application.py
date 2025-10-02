import os
import re
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
import pdfplumber

MODEL_PATH = "guilhermesumita000/email-classifier"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_email(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_idx = torch.argmax(probs, dim=1).item()
    label = model.config.id2label[label_idx]  # pega o nome da classe
    return label

def extract_text_from_pdf(file):
    """Extrai texto de PDF"""
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
        if text:
            return text
    except Exception:
        pass
    try:
        with pdfplumber.open(file) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
        return text
    except Exception:
        return ""

def classify_email(email_text, email_file=None):
    text = email_text
    if email_file:
        file_ext = email_file.name.lower().split('.')[-1]
        if file_ext == "pdf":
            text_from_file = extract_text_from_pdf(email_file)
            if text_from_file:
                text = text_from_file
            else:
                return "Erro: PDF contém apenas imagens ou formato não suportado", ""
        elif file_ext == "txt" and not text:
            text = email_file.read().decode("utf-8")

    if not text or not text.strip():
        return "Erro: Nenhum texto fornecido", ""

    label = predict_email(text)

    if label.lower() == "produtivo":
        suggested = (
            f"Olá, obrigado pelo contato. Recebemos sua solicitação e vamos analisar. "
            f"Encaminhei ao time responsável — retornaremos com atualização em até 48h.\n\n"
            f"Resumo do pedido: {text[:250]}"
        )
    else:
        suggested = (
            "Olá! Obrigado pela mensagem — mensagem registrada. "
            "Se for necessário alguma ação, por favor reenvie com mais detalhes."
        )

    # CORREÇÃO: retorna dois valores separados
    return label, suggested


# --- Interface Gradio ---
iface = gr.Interface(
    fn=classify_email,
    inputs=[
        gr.Textbox(label="Texto do e-mail", placeholder="Cole o e-mail aqui..."),
        gr.File(label="Arquivo PDF ou TXT (opcional)", file_types=[".pdf", ".txt"], type="filepath")
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Categoria"),
        gr.Textbox(label="Resposta Sugerida")
    ],
    title="Classificador de E-mails",
    description="Classifica e-mails em Produtivo ou Não Produtivo e sugere resposta automática.",
)

if __name__ == "__main__":
    iface.launch()
