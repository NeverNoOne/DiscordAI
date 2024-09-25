import discord
from discord.ext import commands
import constants

TOKEN = constants.BotToken()
GUILD_ID = constants.Naumberg_GuildID()
VOICE_CHANNEL_ID = constants.Naumberg_VCID()
TEXTCHANNEL_ID = "710092876176162900"
USER_ID = ""
FILE_PATH = "M:/AI_DataSets/Discord/Naumberg/dies_das.txt"

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
                with open(FILE_PATH, 'w', encoding='utf-8') as f:
                    counter = 0
                    for m in messages:
                        try:
                            if m.author.bot:
                                continue
                            if not m.content.startswith('http'): #and m.author.id == int(USER_ID):
                                counter +=1
                                f.write(f"{m.created_at},{m.author.name},{len(m.attachments)>0},<START> {m.content} <END>\n")
                        except:
                            print(f"error at {m.content}")
                    print(f"finished - found: {counter}")


bot.run(TOKEN)