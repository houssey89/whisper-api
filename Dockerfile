FROM python:3.10-slim

# Installer FFmpeg pour Whisper
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton application Flask
COPY server.py .

# Exposer le port 8080 requis par Cloud Run
ENV PORT=8080

# Commande de démarrage
CMD ["python", "server.py"]
