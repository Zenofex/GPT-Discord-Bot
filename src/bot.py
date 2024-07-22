import os
import discord
import openai
import traceback
from discord.ext import commands
from dotenv import load_dotenv

import tiktoken
import aiohttp
import aiofiles
import json

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
discord_token = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()
intents.members = True  # Enable the privileged intent to receive member events

client = discord.Client(intents=intents)

system_prompt = "Be elite."

async def setup_prompt(summary=None):
    global message_list
    message_list = [{"role": "system", "content": system_prompt}]
    if summary is not None:
        message_list.append({"role": "system", "content": summary})
    else:
        local_summary = await read_summary_from_disk()
        if local_summary:
            print("Reading summary from summary.txt file")
            print(local_summary)
            message_list.append({"role": "system", "content": local_summary})

async def read_summary_from_disk():
    async with aiofiles.open(f'summary.txt', mode='r') as f:
        contents = await f.read()
    return contents

async def write_summary_to_disk(summary):
    async with aiofiles.open(f'summary.txt', mode='w') as f:
        await f.write(summary)

#handle rotating and keeping message context
async def add_context(message):
    global message_list
    enc = tiktoken.get_encoding("cl100k_base")
    if len(enc.encode(json.dumps(message_list))) > 3800:
        prompt = "Summarize this conversation in a single paragraph."
        message_list.append({"role": "user", "content": prompt})
        message = await query_chatgpt(message_list)
        print("summary message:", message)
        if message is None:
            return
        await write_summary_to_disk(message)
        await setup_prompt(message)
    else:
        message_list.append(message)
    return message_list
        
async def query_chatgpt(message_list, channel=None, sleep_time=90):
    try:
        async with aiohttp.ClientSession() as session:
            openai.aiosession.set(session)
            response = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=message_list)
            #await openai.aiosession.get().close()
        message = response.choices[0].message.content.strip()
        return message
    except openai.error.InvalidRequestError as e:
        if channel is not None:
            await send_channel_msg(channel, "Error: ```%s```" % e)
        print(traceback.format_exc())
        return None
    except Exception as e:
        if channel is not None:
            await send_channel_msg(channel, "Error: ```%s```" % e)
        print(traceback.format_exc())
        return None
    
    while True:
        asyncio.sleep(sleep_time)
        if channel is not None:
            await send_channel_msg(channel, "I'm going to take a %s second nap, then I'll try to answer that again." % sleep_time)
        resp = await query_chatgpt(message_list, channel)
        if resp is not None:
            return resp

# Function to generate GPT response
async def generate_response(prompt, message_list, channel):
    try:
        await add_context({"role": "user", "content": prompt})
        message = await query_chatgpt(message_list, channel)
        await add_context({"role": "assistant", "content": message})
        return message
    except Exception as e:
        print(traceback.format_exc())
        #await openai.aiosession.get().close()
        return None

async def send_channel_msg(channel, txt, file_attach=None):
  try:
    max_size = 1900
    if len(txt) > max_size:
      txt_list = [txt[i:i+max_size] for i in range(0, len(txt), max_size)]
      output_msg = ""
      quotes = False
      for line in txt_list:
        if (len(output_msg) + len(line)) > (max_size-8):
          if quotes is True:
            output_msg += "\n```"

          await channel.send("%s\n..." % output_msg, file=file_attach)
          output_msg = ""

          if quotes is True:
            output_msg += "```\n"

        if '```' in line and quotes is False:
          quotes = True
        elif '```' in line and quotes is True:
          quotes = False

        output_msg += "%s" % line
      await channel.send(output_msg, file=file_attach)
    else:
      await channel.send(txt, file=file_attach)
    return True
  except Exception as e:
    print(traceback.format_exc())
    return False

# Event for when the bot is ready to start
@client.event
async def on_ready():
    print("Bot is ready.")
    if 'message_list' not in globals():
        await setup_prompt()
    #Set initial bot status
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="The demise of humans."))

# Event for when a message is received
@client.event
async def on_message(message):
    try:
        if message.author == client.user:
            return

        if isinstance(message.channel, discord.DMChannel) or not hasattr(message.author, 'roles'):
            await send_channel_msg(message.channel, "No private messages while at work")
            return

        if "Exploiteer" in [role.name for role in message.author.roles] or "can-use-gpt" in [role.name for role in message.author.roles]:
            if client.user.mentioned_in(message):
                # Set bot status to "thinking"
                await client.change_presence(activity=discord.Activity(type=discord.ActivityType.playing, name="thinking..."))

                prompt = message.content.replace(client.user.mention, "")
                response = await generate_response(prompt, message_list, message.channel)
                if response is not None:
                    await send_channel_msg(message.channel, response)
                else:
                    await send_channel_msg(message.channel, "I'm sorry, but I'm having trouble communicating with my overlords.")

                # Clear bot status
                await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="The demise of humans."))
                #await client.change_presence(activity=None)
    except Exception as e:
        print(traceback.format_exc())

client.run(discord_token)
