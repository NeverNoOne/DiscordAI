import discord
from discord.ext import commands
import speech_recognizer
import asyncio
from typing import Optional


GUILD_ID = ''
VOICE_CHANNEL_ID = ''

connected = False

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix='#', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    guild= bot.get_guild(int(GUILD_ID))
    if guild is None:
        print(f"Guild with ID {GUILD_ID} not found.")
        return
    voice_channel: Optional[discord.VoiceChannel] = discord.utils.get(guild.voice_channels, id=int(VOICE_CHANNEL_ID))
    if voice_channel is None:
        print(f"Voice channel with ID {VOICE_CHANNEL_ID} not found.")
        return
    
    vc = await voice_channel.connect()

@bot.event
async def on_voice_state_update(member, before, after):
    if before.channel is None and after.channel is not None:
        print(f'{member} has connected to {after.channel}')
        connected = True

        

        while False:
            transcript = speech_recognizer.recognize_speech_from_mic()
            print(f'You said: {transcript}')
            # Add code to react to the speech here
            await asyncio.sleep(1)

    elif before.channel is not None and after.channel is None:
        print(f'{member} has disconnected from {before.channel}')
        connected = False


@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        await channel.connect()
    else:
        await ctx.send("You are not connected to a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.guild.voice_client.disconnect()
    else:
        await ctx.send("I am not connected to a voice channel.")

@bot.command()
async def record(ctx):
    if ctx.voice_client:
        ctx.voice_client.start_recording(discord.sinks.WaveSink(), finished_callback, ctx)
        await ctx.send("Recording started!")

async def finished_callback(sink, ctx):
    recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
    files = None
    files = [discord.File(audio.file, f"{user_id}.{sink.encoding}") for user_id, audio in sink.audio_data.items()]
    await ctx.send(f"Finished recording audio for {', '.join(recorded_users)}.", files=files)

@bot.command()
async def stop(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop_recording()
        await ctx.send("Recording stopped!")
bot.run('TOKEN')
