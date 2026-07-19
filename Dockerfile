FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY baseprompt.json ./baseprompt.json

ENV QODER_HOST=0.0.0.0 \
    QODER_PORT=8963

EXPOSE 8963

CMD ["python", "-u", "openai_bridge.py"]
