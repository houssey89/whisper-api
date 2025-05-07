from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import requests
import os

app = Flask(__name__)
model = WhisperModel("tiny")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPABASE_CHAT_URL = os.environ.get("SUPABASE_CHAT_URL")

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
            return f"[Erreur traduction: {response.status_code}]"
    except Exception as e:
        return f"[Erreur traduction: {str(e)}]"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    user_id = request.form.get('userId') or (request.json.get('userId') if request.is_json else None)
    lat = request.form.get('lat') or (request.json.get('lat') if request.is_json else None)
    lng = request.form.get('lng') or (request.json.get('lng') if request.is_json else None)

    text = ""
    lang_code = "fr"

    # 🔊 Transcription audio
    if 'file' in request.files:
        audio_file = request.files['file']
        segments, info = model.transcribe(audio_file)
        text = " ".join([seg.text for seg in segments])
        lang_code = info.language or "fr"

    # 💬 Texte brut (ex: Voiceflow)
    elif request.is_json:
        body = request.get_json()
        text = body.get("text", "")
        lang_code = body.get("lang", "fr")  # si détecté côté client
    else:
        return jsonify({"error": "Aucun fichier audio ou texte fourni"}), 400

    # 🇫🇷 Traduction
    texte_fr = traduire(text, from_lang=lang_code, to_lang="fr") if lang_code != "fr" else text

    # 🤖 Appel Supabase Function
    try:
        chat_payload = {
            "text": texte_fr,
            "userId": user_id,
            "lat": float(lat) if lat else None,
            "lng": float(lng) if lng else None
        }
        response = requests.post(SUPABASE_CHAT_URL, json=chat_payload)
        if response.ok:
            reponse_fr = response.json().get("answer", "Désolé, pas de réponse.")
        else:
            reponse_fr = "[Erreur fonction chat]"
    except Exception as e:
        reponse_fr = f"[Erreur API chat: {str(e)}]"

    reponse_finale = traduire(reponse_fr, from_lang="fr", to_lang=lang_code) if lang_code != "fr" else reponse_fr

    return jsonify({
        "langue_detectee": lang_code,
        "transcription": text,
        "texte_fr": texte_fr,
        "reponse": reponse_finale
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



