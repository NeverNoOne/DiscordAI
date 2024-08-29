import discord
from discord.ext import commands
import constants

BOTTOKEN = constants.BotToken()
GUILD_ID = int(constants.Test_GuildID())
VC_ID = int(constants.Test_VCID())

intents = discord.Intents.default()
intents.voice_states = True

bot = commands.Bot(command_prefix='#', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_voice_state_update(member:discord.Member, before:discord.VoiceState, after:discord.VoiceState):
    after.channel
    

bot.run(BOTTOKEN)