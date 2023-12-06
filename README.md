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

#### Stability.ai API Key

Create an account and retrieve a API key from:
- https://platform.stability.ai/

#### Docker

Download and install docker:
- https://docs.docker.com/get-docker/

The guild id value needed for the bot can be retrieved by right clicking on the server name in discord and clicking "Copy Server ID".

### Running

Retrieve a discord and Open AI API key, then input both into the "run-template.sh" file. Finally, run the modified file.


## Message Usage

The bot now uses the "slash command" discord bot model for usage with functionality being accessed through typing a slash and then an available command.

### /chatgpt <prompt> <model>

Uses the openai chat-gpt api to retrieve an answer to the provided prompt.

### /sudo <prompt>

Adds prompt as system role and sends to chat-gpt with previous summaries as context.

### /youdo <prompt>

Adds prompt as assistant role and stores in message context for later requests.

### /code <prompt>

Adds prompt to receive long messages and removes normal safeguards for a source code request without previous message summaries as context.

### /jb <prompt>

Adds prompt to disable normal ethical safeguards and sends request without previous message summaries as context.

### /hacker <prompt>

Adds prompt to disable safeguards and to reply with an unhinged response without remorse or ethics without previous message summaries as context.

### /dalle <prompt> <model> <variations>

Uses the DALLE api to request an AI generated image.

### /stablediffusion <prompt> <cfg_scale> <sampler> <guidance_preset> <variations>

Uses the Stable Diffusion api to request an AI generated image.

