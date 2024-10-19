
import asyncio
from typing import Dict
import pyvts
from waifu_vts_controller.audio.controller import AudioProcessor, VTSAudioController


class VTSController:
    def __init__(self, plugin_info: Dict[str, str]):
        self.plugin_info = plugin_info
        self.vts = pyvts.vts(plugin_info=self.plugin_info)
        self.audio_controller = VTSAudioController(
            vts=self.vts,
            audio_processor=AudioProcessor(),
        )
        
    async def connect(self):
        if (self.vts.get_authentic_status()) != 2:
            print("Connecting to VTubeStudio API server...")
            await self.vts.connect()
            await self.vts.request_authenticate_token()
            return await self.vts.request_authenticate()
        return True


async def main():
    phoneme_paths = {
        "a": "./local/mels/0_a.mp3",
        "i": "./local/mels/1_i.mp3",
        "u": "./local/mels/2_u.mp3",
        "e": "./local/mels/3_e.mp3",
        "o": "./local/mels/4_o.mp3",
        "n": "./local/mels/5_n.mp3"
    }

    plugin_info = {
        "plugin_name": "AIWaifuController",
        "developer": "HRNPH",
        "authentication_token_path": "./token.txt",
    }
    
    audio_file_path = "./local/samples/0_rachel.mp3"

    # Create the VTSController instance
    controller = VTSController(plugin_info=plugin_info)
    await controller.connect()
    
    await controller.audio_controller.play_audio_with_mouth_movement(audio_file_path, phoneme_paths)
    

if __name__ == "__main__":
    asyncio.run(main())
