FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gdown

COPY src/   ./src/
COPY api/   ./api/

# Créer les dossiers vides
RUN mkdir -p models data/processed data/raw

# Script de téléchargement des modèles
COPY download_models.py .

# Télécharger les modèles depuis Google Drive au build
RUN python download_models.py

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]