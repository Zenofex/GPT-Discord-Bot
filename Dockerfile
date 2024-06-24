FROM ubuntu:latest

RUN apt-get update && apt-get install libgl1 git libglib2.0-0 python3 python3-pip -y

WORKDIR /app

COPY requirements.txt .

ENV DISCORD_TOKEN=<DISCORD_TOKEN>

ENV OPENAI_API_KEY=<OPENAI_API_KEY>

ENV SUMMARY_FILE=<SUMMARY_FILE>

RUN pip3 install -r requirements.txt --break-system-packages

VOLUME /app/data/

COPY src/ /src/

CMD ["python3", "/src/bot.py"]
