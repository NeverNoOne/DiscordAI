from discord.abc import Connectable
import websockets
import json
import asyncio
import constants
import select
import discord
from discord import utils
from discord.gateway import DiscordWebSocket, DiscordVoiceWebSocket
from discord.voice_client import VoiceClient

GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
TOKEN = constants.BotToken()
#TODO implement logic for starting and ending the audio receiving and test if it actually receives 

#TODO possible without joining vc???

#TODO cleanup

class CustomDiscordVoiceWebSocket(discord.voice_client.DiscordVoiceWebSocket):
    async def received_message(self, msg):
        op = msg.get('op')
        if op == 5:
            self.handle_speaking_event(msg['d'])

        await super().received_message(msg)

    def handle_speaking_event(self, data):
        user_id = data['user_id']
        speaking = data['speaking']
        print(f"User {user_id} is speaking: {speaking}")
    
class CustomVoiceClient(VoiceClient):
    def __init__(self, client: discord.Client, channel: Connectable):
        super().__init__(client, channel)
        self.listening = False

    async def connect_websocket(self) -> DiscordVoiceWebSocket:
        ws = await CustomDiscordVoiceWebSocket.from_client(self)
        self.ws = ws
        return await super().connect_websocket()

    async def recv_audio_data(self):
        if self.socket:
            while self.listening:
                ready, _, err = select.select([self.socket], [], [self.socket], 0.01)
                if not ready:
                    print(f'Socket error: {err}')
                    continue

                try:
                    data = self.socket.recv(4096)
                except OSError:
                    self.listening = False
                    continue
                print(data)

    async def start_listening(self):
        self.listening = True
        await self.recv_audio_data()

class CustomDiscordWebSocket(DiscordWebSocket):
    async def received_message(self, msg):
        if type(msg) is bytes:
            self._buffer.extend(msg)

            if len(msg) <4 or msg[-4:] != b'\x00\x00\xff\xff':
                return
            msg = self._zlib.decompress(self._buffer)
            msg = msg.decode("utf-8")
            self._buffer = bytearray()
        
        self.log_receive(msg)
        msg = utils._from_json(msg)

        print(f'after {msg}')

        event_type = msg.get('t')
        print(event_type)
        if event_type == 'VOICE_STATE_UPDATE':
            self.handle_voice_state_changed(msg['d'])

        tmp = utils._to_json(msg)        
        await super().received_message(tmp)
        self.HEARTBEAT

    def handle_voice_state_changed(self, data):
        user_id = data['user_id']
        channel_id = data['channel_id']
        guild_id = data['guild_id']
        # Additional custom logic to handle voice state updates
        print(f"User {user_id} changed voice state in guild {guild_id}, channel {channel_id}")

class CustomBot(discord.Bot):
    async def connect(self, *, reconnect: bool = True) -> None:
        ws = await CustomDiscordWebSocket.from_client(self)
        self.ws = ws
        print(self.ws)
        await super().connect()
        
intents = discord.Intents.default()
intents.voice_states = True

client = CustomBot(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')    
    guild = client.get_guild(int(constants.Naumberg_GuildID()))
    if guild is None: return
    for channel in guild.voice_channels:
        if channel.id == int(constants.Naumberg_VCID()):
            vc = await channel.connect(cls=CustomVoiceClient)
            await vc.start_listening()

#@client.event
#async def on_voice_state_update(member, before, after):
#    if before.channel is None and after.channel is not None:
#        voice_channel = after.channel
#        voice_client = await voice_channel.connect(cls=CustomVoiceClient)
#        await voice_client.recv_audio_data()


client.run(TOKEN)