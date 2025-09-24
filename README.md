# VibeVoice ComfyUI Nodes with exl3 support and realtime speed

A comprehensive ComfyUI integration for Microsoft's VibeVoice text-to-speech model, enabling high-quality single and multi-speaker voice synthesis directly within your ComfyUI workflows.

Original vibevoice-7b works like this: 2 LLM passes (positive+negative) + diffusion based on these two passes. 

## Optimizations
- i took original comfy nodes from https://github.com/Enemyx-net/VibeVoice-ComfyUI
- i replaced LLM engine from HF-transformers to exllamav3 (now LLM stage is 3 times faster)
- i replaced 2 passes of LLM with one, but with a cache for the negative pass
- reduced the number of steps to 5 (the fewer the steps, the less variability)
- added paragraph split for input text (vibevoice starts to glitch on long text)
- I haven't touched diffusion yet, think there is an option to attach some TeaCache
- added streaming playback with 1s buffer. Playback now starts almost instantly


## Requirements:
- nvidia 3000+ (the 2000 series is not compatible with exllamav3, but the node can be run without exllamav3)
- at least 8 GB vram (preferably 12 GB)
- flash-attention-2 (exllamav3 does not work without it)
- my modified exllamav3
- models must be fully loaded into vram, there is no partial offloading

## VRAM consumption and speed:
- 7b-exl-8bit + no-llm-bf16 - 12.6GB
- 7b-exl-4bit + no-llm-bf16 - 9.5GB (realtime at 3090, 9.00 it/s)
- 7b-exl-4bit + no-llm-nf4 - 7.3GB (nf4 is 1.5 times slower)
- 1.5b-exl-8bit + no-llm-nf4 - 4.7GB

- qll exl3 quants run the same speed. But 4bit is a little faster. 
- the nvidia 3060 is only 20% slower than the 3090.

## Installation
- Windows: install precompiled flash-attention-2 and exllamav3 then install these comfy nodes
- Linux: just install as comfyUI nodes

flash-attention-2
It's difficult to compile under Windows, so here are the links for the compiled whl for flash-attention-2:
here https://huggingface.co/lldacing/flash-attention-windows-wheel/tree/main

You can find out your version of Python, Torch, cuda in comfyui - menu - Help - about

Below, I'm using python 3.11, torch 2.6.0, and cuda126. For other versions, please refer to the links above (or compile yourself). For flash-attention, it's important to match the version of python, torch, and cuda. For exllama, the main requirement is that the version of python matches. If you can't find any suitable compiled versions of flash-attention, you can compile them yourself using the following guide: https://www.reddit.com/r/Oobabooga/comments/1jq3uj9/guide_getting_flash_attention_2_working_on/

exllamav3-v0.0.6 - choose whl based on your python version: https://github.com/mozer/exllamav3/releases/tag/v0.0.6

```
cd C:\DATA\SD\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\python_embeded
python.exe -m pip install https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%2Bcu126torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl
# uninstall existing exllamav3, if you have one
python.exe -m pip uninstall exllamav3

# install my exllamav3 for Python 3.11.x (choose correct whl here from my repo, link above)
python.exe -m pip install https://github.com/Mozer/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.7.0-cp311-cp311-win_amd64.whl```
python.exe -m pip install -U triton-windows<3.5
```



After that, install my nodes using the comfyui manager - install via git url:
https://github.com/mozer/comfyUI-vibevoice-exl3
Or via: `cd ComfyUI/custom_nodes && git clone https://github.com/mozer/comfyUI-vibevoice-exl3`
Restart comfyui.

## Using

Workflow with wav2lip (wav2lip is optional): https://github.com/Mozer/ComfyUI-VibeVoice-exl3/blob/main/examples/vibevoice_exl3_with_wav2lip.json
You don't need to download the models manually. They are dowloaded automatically. But if you really want to, they're available here: https://huggingface.co/collections/tensorbanana/vibevoice-68cd1bac5766dc65e90380c1
 If you're going to upload them manually, make sure to study the folder structure first (HF-downloader uses this method). example: /models/vibevoice/models--tensorbanana--vibevoice-1.5b-exl3-8bit/snapshots/badfbb16dd63a1a8e633ba6eb138a21303ed1325/model.safetensors

- You need to load 2 models at once into the node, example: VibeVoice-7B-no-llm-bf16 (3.2GB) + vibevoice-7b-exl3-4bit (4.4 GB). In exl3 quants there is no diffusion, and in no-llm there is no LLM.
- If you have noise in the audio output, reduce the value of negative_llm_steps_to_cache to 1-2 or even to 0 (as in the original, but it will be slower). The longer the chunk, the more likely there will be noise. 
- Use split_by_newline:True to split the text into paragraphs. I do not recommend splitting into sentences, as the intonations will vary in each sentence.







## OLD Readme (left for )

### Core Functionality
- 🎤 **Single Speaker TTS**: Generate natural speech with optional voice cloning
- 👥 **Multi-Speaker Conversations**: Support for up to 4 distinct speakers
- 🎯 **Voice Cloning**: Clone voices from audio samples
- 📝 **Text File Loading**: Load scripts from text files

### Model Options
- 🚀 **Two Model Sizes**: 1.5B (faster) and 7B (higher quality)
- 🔧 **Flexible Configuration**: Control temperature, sampling, and guidance scale

### Performance & Optimization
- ⚡ **Attention Mechanisms**: Choose between auto, eager, sdpa, or flash_attention_2
- 🎛️ **Diffusion Steps**: Adjustable quality vs speed trade-off (default: 20)
- 💾 **Memory Management**: Toggle automatic VRAM cleanup after generation
- 🧹 **Free Memory Node**: Manual memory control for complex workflows

## Video Demo
<p align="center">
  <a href="https://www.youtube.com/watch?v=fIBMepIBKhI">
    <img src="https://img.youtube.com/vi/fIBMepIBKhI/maxresdefault.jpg" alt="VibeVoice ComfyUI Wrapper Demo" />
  </a>
  <br>
  <strong>Click to watch the demo video</strong>
</p>


## Available Nodes

### 1. VibeVoice Load Text From File
Loads text content from files in ComfyUI's input/output/temp directories.
- **Supported formats**: .txt
- **Output**: Text string for TTS nodes

### 2. VibeVoice Single Speaker
Generates speech from text using a single voice.
- **Text Input**: Direct text or connection from Load Text node
- **Models**: VibeVoice-1.5B or VibeVoice-7B-Preview
- **Voice Cloning**: Optional audio input for voice cloning
- **Parameters** (in order):
  - `text`: Input text to convert to speech
  - `model`: VibeVoice-1.5B or VibeVoice-7B-Preview
  - `attention_type`: auto, eager, sdpa, or flash_attention_2 (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `voice_to_clone`: Audio input for voice cloning
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 3. VibeVoice Multiple Speakers
Generates multi-speaker conversations with distinct voices.
- **Speaker Format**: Use `[N]:` notation where N is 1-4
- **Voice Assignment**: Optional voice samples for each speaker
- **Recommended Model**: VibeVoice-7B-Preview for better multi-speaker quality
- **Parameters** (in order):
  - `text`: Input text with speaker labels
  - `model`: VibeVoice-1.5B or VibeVoice-7B-Preview
  - `attention_type`: auto, eager, sdpa, or flash_attention_2 (default: auto)
  - `free_memory_after_generate`: Free VRAM after generation (default: True)
  - `diffusion_steps`: Number of denoising steps (5-100, default: 20)
  - `seed`: Random seed for reproducibility (default: 42)
  - `cfg_scale`: Classifier-free guidance (1.0-2.0, default: 1.3)
  - `use_sampling`: Enable/disable deterministic generation (default: False)
- **Optional Parameters**:
  - `speaker1_voice` to `speaker4_voice`: Audio inputs for voice cloning
  - `temperature`: Sampling temperature (0.1-2.0, default: 0.95)
  - `top_p`: Nucleus sampling parameter (0.1-1.0, default: 0.95)

### 4. VibeVoice Free Memory
Manually frees all loaded VibeVoice models from memory.
- **Input**: `audio` - Connect audio output to trigger memory cleanup
- **Output**: `audio` - Passes through the input audio unchanged
- **Use Case**: Insert between nodes to free VRAM/RAM at specific workflow points
- **Example**: `[VibeVoice Node] → [Free Memory] → [Save Audio]`

## Multi-Speaker Text Format

For multi-speaker generation, format your text using the `[N]:` notation:

```
[1]: Hello, how are you today?
[2]: I'm doing great, thanks for asking!
[1]: That's wonderful to hear.
[3]: Hey everyone, mind if I join the conversation?
[2]: Not at all, welcome!
```

**Important Notes:**
- Use `[1]:`, `[2]:`, `[3]:`, `[4]:` for speaker labels
- Maximum 4 speakers supported
- The system automatically detects the number of speakers from your text
- Each speaker can have an optional voice sample for cloning

## Model Information

### VibeVoice-1.5B
- **Size**: ~5GB download
- **Speed**: Faster inference
- **Quality**: Good for single speaker
- **Use Case**: Quick prototyping, single voices

### VibeVoice-7B-Preview
- **Size**: ~17GB download
- **Speed**: Slower inference
- **Quality**: Superior, especially for multi-speaker
- **Use Case**: Production quality, multi-speaker conversations

Models are automatically downloaded on first use and cached in `ComfyUI/models/vibevoice/`.

## Generation Modes

### Deterministic Mode (Default)
- `use_sampling = False`
- Produces consistent, stable output
- Recommended for production use

### Sampling Mode
- `use_sampling = True`
- More variation in output
- Uses temperature and top_p parameters
- Good for creative exploration

## Voice Cloning

To clone a voice:
1. Connect an audio node to the `voice_to_clone` input (single speaker)
2. Or connect to `speaker1_voice`, `speaker2_voice`, etc. (multi-speaker)
3. The model will attempt to match the voice characteristics

**Requirements for voice samples:**
- Clear audio with minimal background noise
- Minimum 3–10 seconds. Recommended at least 30 seconds for better quality
- Automatically resampled to 24kHz

## Tips for Best Results

1. **Text Preparation**:
   - Use proper punctuation for natural pauses
   - Break long texts into paragraphs
   - For multi-speaker, ensure clear speaker transitions

2. **Model Selection**:
   - Use 1.5B for quick single-speaker tasks
   - Use 7B for multi-speaker or when quality is priority

3. **Seed Management**:
   - Default seed (42) works well for most cases
   - Save good seeds for consistent character voices
   - Try random seeds if default doesn't work well

4. **Performance**:
   - First run downloads models (5-17GB)
   - Subsequent runs use cached models
   - GPU recommended for faster inference

## System Requirements

### Hardware
- **Minimum**: 8GB VRAM for VibeVoice-1.5B
- **Recommended**: 16GB+ VRAM for VibeVoice-7B
- **RAM**: 16GB+ system memory

### Software
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)
- ComfyUI (latest version)

## Troubleshooting

### Installation Issues
- Ensure you're using ComfyUI's Python environment
- Try manual installation if automatic fails
- Restart ComfyUI after installation

### Generation Issues
- If voices sound unstable, try deterministic mode
- For multi-speaker, ensure text has proper `[N]:` format
- Check that speaker numbers are sequential (1,2,3 not 1,3,5)

### Memory Issues
- 7B model requires ~16GB VRAM
- Use 1.5B model for lower VRAM systems
- Models use bfloat16 precision for efficiency

## Examples

### Single Speaker
```
Text: "Welcome to our presentation. Today we'll explore the fascinating world of artificial intelligence."
Model: VibeVoice-1.5B
cfg_scale: 1.3
use_sampling: False
```

### Two Speakers
```
[1]: Have you seen the new AI developments?
[2]: Yes, they're quite impressive!
[1]: I think voice synthesis has come a long way.
[2]: Absolutely, it sounds so natural now.
```

### Four Speaker Conversation
```
[1]: Welcome everyone to our meeting.
[2]: Thanks for having us!
[3]: Glad to be here.
[4]: Looking forward to the discussion.
[1]: Let's begin with the agenda.
```

## Known Limitations

- Maximum 4 speakers in multi-speaker mode
- Works best with English and Chinese text
- Some seeds may produce unstable output
- Background music generation cannot be directly controlled

## License

This ComfyUI wrapper is released under the MIT License. See LICENSE file for details.

**Note**: The VibeVoice model itself is subject to Microsoft's licensing terms:
- VibeVoice is for research purposes only
- Check Microsoft's VibeVoice repository for full model license details

## Links

- [Original VibeVoice Repository](https://github.com/microsoft/VibeVoice) - Official Microsoft VibeVoice repository

## Credits

- **VibeVoice Model**: Microsoft Research
- **ComfyUI Integration**: Fabio Sarracino
- **Base Model**: Built on Qwen2.5 architecture

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review ComfyUI logs for error messages
3. Ensure VibeVoice is properly installed
4. Open an issue with detailed error information

## Contributing

Contributions welcome! Please:
1. Test changes thoroughly
2. Follow existing code style
3. Update documentation as needed

4. Submit pull requests with clear descriptions









