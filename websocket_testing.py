from discord.abc import Connectable
import constants
import select
import discord
import threading
import wave
import time
from speech_recognizer import recognize_speech_from_bytes, recognize_speech_from_wav
from discord import utils
from discord.gateway import DiscordWebSocket, DiscordVoiceWebSocket
from discord.voice_client import VoiceClient
from discord.ext import commands
import discord.opus

GATEWAY = "wss://gateway.discord.gg/?v=10&encoding=json"
TOKEN = constants.BotToken()
#TODO implement logic for starting and ending the audio receiving and test if it actually receives 
#TODO decoding the received audiobytes
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
        self.decoded_audio_bytes:bytes = b''
        self.custom_decoder = discord.opus.Decoder()
        self.wait_interval = 0

    async def connect_websocket(self) -> DiscordVoiceWebSocket:
        ws = await CustomDiscordVoiceWebSocket.from_client(self)
        self.ws = ws
        return await super().connect_websocket()

    def recv_audio_data(self):
        if self.socket:
            
            while self.listening:
                ready, _, err = select.select([self.socket], [], [self.socket], 0.01)
                if not ready:
                    #print(f'Socket error: {err}')
                    continue

                try:
                    data = self.socket.recv(4096)
                except OSError as ex:
                    self.listening = False
                    print(ex)
                    continue
                self.process_audio(data)
                if self.wait_interval > 5 and self.decoded_audio_bytes != b'':
                    print('stopped speaking')
                    print('recognizing speech...')
                    stt = recognize_speech_from_bytes(self.decoded_audio_bytes)
                    print(stt)
                    self.decoded_audio_bytes = b''
                    
            #print('writing wave file...')
            #with wave.open('test.wav', 'wb') as f:
            #    f.setnchannels(self.custom_decoder.CHANNELS)
            #    f.setsampwidth(2)
            #    f.setframerate(48000)
            #    f.writeframes(self.decoded_audio_bytes)
            #print('wave file written and saved')
            
            

    def process_audio(self, data):
        if 200 <= data[1] <= 204:
            # RTCP received.
            # RTCP provides information about the connection
            # as opposed to actual audio data, so it's not
            # important at the moment.
            self.wait_interval += 1
            #print('not important')
            return
        self.wait_interval = 0
        RawAudioData = discord.sinks.RawData(data, self)

        if RawAudioData.decrypted_data == b"\xf8\xff\xfe":  # Frame of silence
            return

        try:
            decoded_audio = self.custom_decoder.decode(RawAudioData.decrypted_data)

            self.decoded_audio_bytes += decoded_audio
        except Exception as e:
            print(f'Error decoding audio: {e}')

    def start_listening(self):
        self.listening = True
        t = threading.Thread(
            target=self.recv_audio_data
        )
        t.start()
        #await self.recv_audio_data()

    def stop_listening(self):
        if self.listening:
            self.listening = False

class CustomDiscordWebSocket(DiscordWebSocket):
    async def received_message(self, msg):
        await super().received_message(msg)
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
        await super().received_message(msg)
        self.HEARTBEAT

    def handle_voice_state_changed(self, data):
        user_id = data['user_id']
        channel_id = data['channel_id']
        guild_id = data['guild_id']
        # Additional custom logic to handle voice state updates
        print(f"User {user_id} changed voice state in guild {guild_id}, channel {channel_id}")

class CustomBot(commands.Bot):
    async def connect(self, *, reconnect: bool = True) -> None:
        ws = await DiscordWebSocket.from_client(self)
        self.ws = ws
        print(self.ws)
        await super().connect()
        
intents = discord.Intents.default()
intents.voice_states = True
intents.message_content = True

client = CustomBot(command_prefix='!', intents=intents, auto_sync_commands=True)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    
    guild = client.get_guild(int(constants.Test_GuildID()))
    if guild is None: return
    for channel in guild.voice_channels:
        if channel.id == int(constants.Test_VCID()):
            vc = await channel.connect(cls=CustomVoiceClient)
            vc.start_listening()

@client.command()
async def stop(ctx):
    if ctx.voice_client:
        ctx.voice_client.stop_listening()
        await ctx.send("Recording stopped!")
    #if ctx.author.voice:
    #    user_channel = ctx.author.voice.channel
    #    channel = next(v for v in client.voice_clients if v.channel == user_channel)
    #    if channel:            
    #        await channel.disconnect(force=False)
    #    else:
    #        await ctx.send("Not connected to your vc.")
    #else:
    #    await ctx.send("You are not connected to a voice channel.")

@client.command()
async def hello(ctx):
    await ctx.send('hello')

#commands:list[discord.ApplicationCommand] = [discord.ApplicationCommand(stop)]

#async def sync_guild_commands():
#    guild = discord.Object(int(constants.Test_GuildID()))
#    await client.sync_commands()


client.run(TOKEN)