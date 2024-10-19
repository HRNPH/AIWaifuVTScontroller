# AIwaifuVTSController
VTS Plugin for AIwaifu using python to
- Voice Buffer as input for VTS Lip Sync
- Expression/Animation Control API

# Usage
```python
import asyncio
from waifu_vts_controller import VTSController

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

```