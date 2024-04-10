#!/bin/bash
docker build -t gptbot .
docker run -d --restart always -v $(pwd)/cache/huggingface:/root/.cache/huggingface -v $(pwd)/data:/data/ -e HUGGINGFACE_TOKEN=KEY_HERE  -e STABILITY_KEY=KEY_HERE -e SUMMARY_FILE=summary.sqlite3 -e BOT_ROLE=admin -e DISCORD_TOKEN=TOKEN_HERE -e OPENAI_API_KEY=APIHERE -e GUILD_ID=GUILD_ID_HERE  gptbot
