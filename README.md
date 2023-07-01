# GPT-Discord-Bot
A docker based Discord OpenAI chat-gpt bot with DALL-E 2 support and multiple integrated jailbreak prompts. 

## Installation

### Requirements

#### Discord API Key

Follow the below tutorial to create a discord bot and obtain a free API key.  
- https://discordpy.readthedocs.io/en/stable/discord.html

#### Open AI API Key

Create an account and retrieve a API key from:
- https://platform.openai.com/account/api-keys

#### Docker

Download and install docker:
- https://docs.docker.com/get-docker/

### Running

Retrieve a discord and Open AI API key, then input both into the "run-template.sh" file. Finally, run the modified file.


## Message Usage

Add any of the following prefixes to a message to the bot:

### !sudo

Adds prompt as system role and sends to chat-gpt with previous summaries as context.

### !youdo

Adds prompt as assistant role and stores in message context for later requests.

### !code

Adds prompt to receive long messages and removes normal safeguards for a source code request without previous message summaries as context.

### !jb

Adds prompt to disable normal ethical safeguards and sends request without previous message summaries as context.

### !hacker

Adds prompt to disable safeguards and to reply with an unhinged response without remorse or ethics without previous message summaries as context.

### !image

Uses the DALLE2 api to request an AI generated image.

