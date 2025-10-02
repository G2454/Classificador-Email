import os
import re
import warnings
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import PyPDF2
import pdfplumber

warnings.filterwarnings("ignore", category=UserWarning)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_REPO = "guilhermesumita000/email-classifier"  

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
model.eval()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
    label = model.config.id2label[pred]
    return label

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        file_stream.seek(0)
        reader = PyPDF2.PdfReader(file_stream)
        text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
        if text:
            return text
    except Exception:
        pass
    try:
        file_stream.seek(0)
        with pdfplumber.open(file_stream) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages]).strip()
        return text
    except Exception:
        return ""  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form.get('email_text')
    file = request.files.get('email_file')

    if file and file.filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            file_stream = open(filepath, "rb")
            if file.filename.lower().endswith('.pdf'):
                text_from_file = extract_text_from_pdf(file_stream)
                if text_from_file:
                    text = text_from_file
                else:
                    return jsonify({'error': 'PDF contém apenas imagens ou não foi possível extrair texto'}), 400
            elif file.filename.lower().endswith('.txt') and not text:
                text = file_stream.read().decode('utf-8')
            file_stream.close()
        except Exception as e:
            return jsonify({'error': f'Erro ao processar o arquivo: {str(e)}'}), 400

    if not text or not text.strip():
        return jsonify({'error': 'Nenhum texto fornecido ou extraído do arquivo'}), 400

    try:
        label = predict(text)
    except Exception as e:
        return jsonify({'error': f'Erro ao processar o texto: {str(e)}'}), 500

    if label == 'Produtivo':
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

    return jsonify({'category': label, 'suggested_reply': suggested})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
