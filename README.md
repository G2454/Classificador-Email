# Classificador de E-mails

Este projeto permite classificar e-mails em **Produtivo** ou **Não Produtivo** e gerar uma resposta sugerida automaticamente.

Ele utiliza:

- Hugging Face para o modelo de classificação
- Gradio para a interface web
- PyPDF2 e pdfplumber para extrair texto de PDFs
- `.env` para armazenar o token da API do Hugging Face

---

## Pré-requisitos

- Python 3.10 ou superior
- Pip
- Git (para clonar o repositório, se necessário)
- Um **token de API Hugging Face** com permissões `fine-grained`

---

## Instalação

1. Clone o repositório (ou baixe os arquivos):

```bash
git clone <URL_DO_REPOSITORIO>
cd <PASTA_DO_PROJETO>

```

2. Crie um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash

pip install -r requirements.txt

```

4. Crie um arquivo .env na raiz do projeto com seu token da Hugging Face:
```bash

HUGGINGFACE_API_TOKEN=SEU_TOKEN_AQUI

```

5. Rode localmente

```bash

python application.py

```
