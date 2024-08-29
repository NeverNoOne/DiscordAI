import asyncio
import websockets
import json
import constants

DISCORD_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
TOKEN = constants.BotToken

async def authenticate(ws):
    # Send an IDENTIFY payload
    identify_payload = {
        "op": 2,
        "d": {
            "token": TOKEN,
            "intents": 513,  # Adjust intents as needed
            "properties": {
                "$os": "windows",
                "$browser": "my_bot",
                "$device": "my_bot"
            }
        }
    }
    await ws.send(json.dumps(identify_payload))

async def handle_messages(ws):
    async for message in ws:
        data = json.loads(message)
        
        print(data)  # Handle raw message data here

async def main():
    async with websockets.connect(DISCORD_GATEWAY_URL) as ws:
        await authenticate(ws)
        await handle_messages(ws)

asyncio.run(main())
