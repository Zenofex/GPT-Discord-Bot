#!/bin/bash
docker build -t gptbot .
docker run -d --restart always -v data:/data/ -e SUMMARY_FILE=summary.txt -e BOT_ROLE=admin -e DISCORD_TOKEN=TOKENHERE -e OPENAI_API_KEY=APIHERE gptbot
