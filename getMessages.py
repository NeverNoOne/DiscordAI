import discord
from discord.ext import commands
import constants

TOKEN = constants.BotToken()
GUILD_ID = constants.Naumberg_GuildID()
VOICE_CHANNEL_ID = constants.Naumberg_VCID()
TEXTCHANNEL_ID = ""
USER_ID = ""

connected = False

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

bot = commands.Bot(command_prefix='#', intents=intents)


@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    guild = bot.get_guild(int(GUILD_ID))
    if guild is not None:
        for channel in guild.text_channels:
            if channel.id == int(TEXTCHANNEL_ID):
                print("channel found!")
                messages = [message async for message in channel.history(limit=20000)]
                #print(messages.__len__())
                with open('history.txt', 'w', encoding='utf-8') as f:
                    counter = 0
                    for m in messages:
                        try:
                            if m.author.bot:
                                continue
                            if not m.content.startswith('http') and m.author.id == int(USER_ID):
                                counter +=1
                                f.write(f"<s>{m.content}<e>\n")
                        except:
                            print(f"error at {m.content}")
                    print(f"finished - found: {counter}")


bot.run(TOKEN)