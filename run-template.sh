#!/bin/bash
docker build -t gptbot .
docker run -d --restart always -v data:/data/ -e SUMMARY_FILE=summary.sqlite3 -e BOT_ROLE=admin -e DISCORD_TOKEN=TOKENHERE -e OPENAI_API_KEY=APIHERE gptbot
