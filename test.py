import constants
import discord
import numpy as np
import webrtcvad

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(1)  # 0: least aggressive, 3: most aggressive

    async def on_ready(self):
        print(f'Logged in as {self.user}')

    async def on_message(self, message):
        if message.content.startswith('#join'):
            if message.author.voice:
                channel = message.author.voice.channel
                vc = await channel.connect()
                a = vc.recv_audio(self.AudioSink(self.vad), self.on_Callbalck)
                print(a)
    
    async def on_Callbalck(self):
        print("called")

    #async def on_voice_state_update(self, member, before, after):
        #if after.channel and not before.channel:            

    class AudioSink(discord.sinks.Sink):
        def __init__(self, vad):
            self.vad = vad

        def write(self, data):
            audio_data = np.frombuffer(data, dtype=np.int16)
            is_speech = self.vad.is_speech(audio_data.tobytes(), 16000)
            if is_speech:
                print("Voice detected!")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
client = MyClient(intents=intents)
client.run(constants.BotToken())