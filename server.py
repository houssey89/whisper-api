from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import requests
from gtts import gTTS
import base64
from io import BytesIO
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
model = WhisperModel("tiny")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPABASE_CHAT_URL = os.environ.get("SUPABASE_CHAT_URL")
JULES_API_URL = os.environ.get("JULES_API_URL", "https://api.jules.ai/chat")
MED_MODEL_ID = os.environ.get("MED_MODEL_ID")
clip_processor = CLIPProcessor.from_pretrained(MED_MODEL_ID) if MED_MODEL_ID else None
clip_model = CLIPModel.from_pretrained(MED_MODEL_ID) if MED_MODEL_ID else None

LISTE_PRODUITS = [
    {"nom": "Doliprane", "prix": 2.5},
    {"nom": "Efferalgan", "prix": 3.0},
    {"nom": "Aspirine", "prix": None},
]

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


@app.route("/transcribe_jules", methods=["POST"])
def transcribe_jules():
    """Transcrit l'audio, envoie la transcription à l'API Jules
    puis renvoie la réponse et un audio TTS."""
    text = ""

    if 'file' in request.files:
        audio_file = request.files['file']
        segments, _ = model.transcribe(audio_file)
        text = " ".join([seg.text for seg in segments])
    elif request.is_json:
        body = request.get_json()
        text = body.get("text", "")
    else:
        return jsonify({"error": "Aucun fichier audio ou texte fourni"}), 400

    try:
        resp = requests.post(JULES_API_URL, json={"content": text})
        resp.raise_for_status()
        jules_answer = resp.json().get("answer") or resp.text
    except Exception as e:
        jules_answer = f"[Erreur API Jules: {e}]"

    audio_b64 = None
    try:
        tts = gTTS(text=jules_answer, lang="fr")
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        jules_answer += f" [Erreur TTS: {e}]"

    return jsonify({
        "transcription": text,
        "jules_reponse": jules_answer,
        "audio_base64": audio_b64,
    })


@app.route("/identify_med", methods=["POST"])
def identify_med():
    """Identifie un médicament à partir d'une image."""
    if "file" not in request.files:
        return jsonify({"error": "Aucune image fournie"}), 400
    if clip_processor is None or clip_model is None:
        return jsonify({"produit": None, "prix": None})

    image = Image.open(request.files["file"]).convert("RGB")
    textes = [p["nom"] for p in LISTE_PRODUITS]
    inputs = clip_processor(text=textes, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits = outputs.logits_per_image[0]
    idx = logits.argmax().item()
    produit = LISTE_PRODUITS[idx]["nom"]
    prix = LISTE_PRODUITS[idx].get("prix")
    return jsonify({"produit": produit, "prix": prix})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)



