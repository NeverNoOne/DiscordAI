import discord
import asyncio
import speech_recognizer
from typing import Optional


GUILD_ID = ''
VOICE_CHANNEL_ID = ''

intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')
    guild: Optional[discord.Guild] = discord.utils.get(client.guilds, id=int(GUILD_ID))
    if guild is None:
        print(f"Guild with ID {GUILD_ID} not found.")
        return
    voice_channel: Optional[discord.VoiceChannel] = discord.utils.get(guild.voice_channels, id=int(VOICE_CHANNEL_ID))
    if voice_channel is None:
        print(f"Voice channel with ID {VOICE_CHANNEL_ID} not found.")
        return
    
    vc = voice_channel.connect()

    print(vc)
    # while True:
    #     transcript = speech_recognizer.recognize_speech_from_mic()
    #     print(f'You said: {transcript}')
    #     # Add code to react to the speech here
    #     await asyncio.sleep(1)

client.run('TOKEN')
