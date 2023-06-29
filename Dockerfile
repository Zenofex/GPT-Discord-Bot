FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .

ENV DISCORD_TOKEN=<DISCORD_TOKEN>

ENV OPENAI_API_KEY=<OPENAI_API_KEY>

ENV SUMMARY_FILE=<SUMMARY_FILE>

RUN pip install -r requirements.txt

VOLUME /app/data/

COPY src/ /src/

CMD ["python", "/src/bot.py"]
