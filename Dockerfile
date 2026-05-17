FROM alpine:latest
FROM python:3.12-slim

# Dépendances système nécessaires pour XGBoost et compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le projet
COPY src/              ./src/
COPY api/              ./api/
COPY models/           ./models/
COPY data/processed/   ./data/processed/

# Dossiers nécessaires
RUN mkdir -p data/raw

# Port
EXPOSE 8000

# Lancer l'API depuis la racine du projet
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]