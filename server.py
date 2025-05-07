from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import os

app = Flask(__name__)
model = WhisperModel("tiny")  # modèle plus léger que "base"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['file']
    segments, _ = model.transcribe(audio_file)

    text = " ".join([seg.text for seg in segments])
    return jsonify({"text": text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run impose le port d'écoute
    app.run(host="0.0.0.0", port=port)


