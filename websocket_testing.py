import websockets
import json
import asyncio
import constants

GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
TOKEN = constants.BotToken()