FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

CMD ["python", "server.py"]
