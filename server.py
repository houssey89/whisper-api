from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import requests
import os

# Initialisation de l'app Flask et du modèle de transcription Whisper
app = Flask(__name__)
model = WhisperModel("tiny")

# Récupération de la clé API Google Translate depuis les variables d'environnement
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Fonction de traduction via l'API Google Translate
def traduire(text, from_lang, to_lang):
    if from_lang == to_lang or not text:
        return text

    url = "https://translation.googleapis.com/language/translate/v2"
    params = {
        "q": text,
        "source": from_lang,
        "target": to_lang,
        "format": "text",
        "key": GOOGLE_API_KEY
    }

    try:
        response = requests.post(url, data=params)
        if response.ok:
            return response.json()["data"]["translations"][0]["translatedText"]
        else:
            return f"[Erreur de traduction: {response.status_code}]"
    except Exception as e:
        return f"[Erreur: {str(e)}]"

# Route principale de transcription + traduction
@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['file']

    # Transcription avec Whisper
    segments, info = model.transcribe(audio_file)
    text = " ".join([seg.text for seg in segments])
    lang_code = info.language or "fr"

    # Traduction vers le français si nécessaire
    if lang_code != "fr":
        texte_fr = traduire(text, from_lang=lang_code, to_lang="fr")
    else:
        texte_fr = text

    # Traitement simulé : ici tu peux appeler ton backend ou Supabase
    reponse_fr = "Le médicament est disponible à la pharmacie X."

    # Retraduction vers la langue d'origine
    if lang_code != "fr":
        reponse_finale = traduire(reponse_fr, from_lang="fr", to_lang=lang_code)
    else:
        reponse_finale = reponse_fr

    return jsonify({
        "langue_detectee": lang_code,
        "transcription": text,
        "texte_fr": texte_fr,
        "reponse": reponse_finale
    })

# Démarrage du serveur Cloud Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



