# Created by Fabio Sarracino

import logging
import os
import tempfile
import torch
import numpy as np
import re
import time
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode, BufferedPyAudioStreamer

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceSingleSpeakerNode(BaseVibeVoiceNode):
    def __init__(self):
        super().__init__()
        # Register this instance for memory management
        try:
            from .free_memory_node import VibeVoiceFreeMemoryNode
            VibeVoiceFreeMemoryNode.register_single_speaker(self)
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a test of the VibeVoice text-to-speech system.", 
                    "tooltip": "Text to convert to speech. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (["VibeVoice-1.5B", "VibeVoice-7B", "VibeVoice-1.5B-no-llm-bf16", "VibeVoice-7B-no-llm-bf16"], {
                    "default": "VibeVoice-1.5B", 
                    "tooltip": "Model to use. 1.5B is faster, 7B has better quality"
                }),
                "quantization_mode": (["bf16", "fp16", "bnb_nf4"], {
                    "default": "bf16",
                    "tooltip": "Default is bf16 (fast). bnb_nf4 - for low vram, 2 times slower."
                }),
                "exllama_model": (["None", "vibevoice-1.5b-exl3-8bit", "vibevoice-7b-exl3-8bit", "vibevoice-7b-exl3-6bit", "vibevoice-7b-exl3-4bit", "vibevoice-7b-exl3-3bit"], {
                    "default": "None", 
                    "tooltip": "exllama model to use. Should be used together with no-llm-bf16 model. Models are automatically downloaded to /models/vibevoice/"
                }),
                "attention_type": (["auto", "eager", "sdpa", "flash_attention_2", "sage"], {
                    "default": "auto",
                    "tooltip": "Attention implementation. Auto selects the best available, eager is standard, sdpa is optimized PyTorch, flash_attention_2 requires compatible GPU"
                }),
                "free_memory_after_generate": ("BOOLEAN", {"default": True, "tooltip": "Free model from memory after generation to save VRAM/RAM. Disable to keep model loaded for faster subsequent generations"}),
                "diffusion_steps": ("INT", {"default": 15, "min": 3, "max": 50, "step": 1, "tooltip": "Number of denoising steps. More steps = better quality but slower. Default: 20"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "cfg_scale": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "voice_to_clone": ("AUDIO", {"tooltip": "Optional: Reference voice to clone. If not provided, synthetic voice will be used."}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "streaming": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming mode, playback directly to your default audio device"}),
                "streaming_buffer": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1, "tooltip": "Seconds to buffer before playing when streaming. Default: 5"}),
                "negative_llm_steps_to_cache": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Cache n steps of negative conditioning for speed-up. 0 - cache turned off for best quality; 20 - for speed (may cause noise)"}),
                "increase_cfg": ("BOOLEAN", {"default": False, "tooltip": "Increase CFG +50% for first 50% of diffusion steps (Experimental, for more emotions)"}),
                "split_by_newline": ("BOOLEAN", {"default": True, "tooltip": "Split long text into chunks by newline"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate speech from text using Microsoft VibeVoice with optional voice cloning"

    def _prepare_voice_samples(self, speakers: list, voice_to_clone) -> List[np.ndarray]:
        """Prepare voice samples from input audio or create synthetic ones"""
        
        if voice_to_clone is not None:
            # Use the base class method to prepare audio
            audio_np = self._prepare_audio_from_comfyui(voice_to_clone)
            if audio_np is not None:
                return [audio_np]
        
        # Create synthetic voice samples for speakers
        voice_samples = []
        for i, speaker in enumerate(speakers):
            voice_sample = self._create_synthetic_voice_sample(i)
            voice_samples.append(voice_sample)
            
        return voice_samples    
  

    def generate_speech(self, text: str = "", model: str = "VibeVoice-1.5B", 
                    attention_type: str = "auto", free_memory_after_generate: bool = True,
                    diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                    use_sampling: bool = False, voice_to_clone=None,
                    temperature: float = 0.95, top_p: float = 0.95, quantization_mode = "bf16", 
                    streaming = False, streaming_buffer = 10, exllama_model = "None", 
                    negative_llm_steps_to_cache: int = 0, increase_cfg: bool=False, 
                    split_by_newline: bool=False):
        """Generate speech from text using VibeVoice"""
        
        audio_dict = None
        
        try:
            if not text or not text.strip():
                raise Exception("No text provided. Please enter text or connect from LoadTextFromFile node.")
            
            final_text = text
            model_mapping = self._get_model_mapping()
            model_path = model_mapping.get(model, model)
            self.load_model(model_path, attention_type, quantization_mode, exllama_model)
            
            speakers = ["Speaker 1"]
            sample_rate = 24000
            
            text_chunks = [final_text.strip()]
            if split_by_newline:
                chunks = [chunk.strip() for chunk in final_text.split('\n') if chunk.strip()]
                if chunks:
                    text_chunks = chunks
            
            voice_samples = self._prepare_voice_samples(speakers, voice_to_clone)
            max_new_tokens = 32637
            
            if streaming and split_by_newline:
                audio_streamer = BufferedPyAudioStreamer(sample_rate=sample_rate, buffer_duration=streaming_buffer)
                audio_streamer.start_playback()

                for chunk_text in text_chunks:
                    # --- THIS IS THE FIX ---
                    # Reset the streamer's finished flag before processing the next chunk.
                    # This ensures the new chunk's inference doesn't stop prematurely.
                    if hasattr(audio_streamer, 'finished_flags'):
                        audio_streamer.finished_flags[0] = False

                    self._generate_with_vibevoice(
                        self._format_text_for_vibevoice(chunk_text, speakers), 
                        voice_samples, cfg_scale, seed, diffusion_steps, 
                        use_sampling, temperature, top_p, 
                        audio_streamer=audio_streamer
                    )

                # ... (no changes to the rest of the function) ...
                audio_streamer.close()
                while audio_streamer.playing:
                    time.sleep(0.1)
                
                full_audio_np = audio_streamer.get_full_audio()
                concatenated_waveform = (
                    torch.from_numpy(full_audio_np).unsqueeze(0).unsqueeze(0)
                    if full_audio_np.size > 0
                    else torch.zeros(1, 1, int(0.5 * sample_rate), dtype=torch.float32)
                )
                audio_dict = {"waveform": concatenated_waveform, "sample_rate": sample_rate}

            else: 
                all_audio_segments = []
                for i, chunk_text in enumerate(text_chunks):
                    chunk_audio_dict = self._generate_with_vibevoice(
                        self._format_text_for_vibevoice(chunk_text, speakers), 
                        voice_samples, cfg_scale, seed, diffusion_steps, 
                        use_sampling, temperature, top_p, 
                        streaming=streaming, buffer_duration=streaming_buffer,
                        max_new_tokens=max_new_tokens, 
                        negative_llm_steps_to_cache=negative_llm_steps_to_cache, 
                        increase_cfg=increase_cfg
                    )
                    
                    if i == 0:
                        sample_rate = chunk_audio_dict.get("sample_rate", 24000)
                    
                    waveform = chunk_audio_dict.get("waveform")
                    if waveform is not None and waveform.numel() > 0:
                        all_audio_segments.append(waveform)
                
                if not all_audio_segments:
                    concatenated_waveform = torch.zeros(1, 1, int(0.5 * sample_rate), dtype=torch.float32)
                elif len(all_audio_segments) > 1:
                    concatenated_waveform = torch.cat(all_audio_segments, dim=-1)
                else:
                    concatenated_waveform = all_audio_segments[0]
                
                audio_dict = {"waveform": concatenated_waveform, "sample_rate": sample_rate}
            
            return (audio_dict,)

        except Exception as e:
            import comfy.model_management as mm
            if isinstance(e, mm.InterruptProcessingException):
                logger.info("Generation interrupted by user.")
                raise
            else:
                logger.error(f"Single speaker speech generation failed: {str(e)}")
                self.free_memory()
                # Check for shape mismatch error
                if "shape mismatch" in str(e):
                    raise Exception("Incorrect models are used. Make sure to use 2 corresponding models (7b-no-llm with 7b-exl3 or 1.5b-no-llm with 1.5b-exl3)")
                else:
                    raise Exception(f"Error generating speech: {str(e)}")
                
        finally:
            if free_memory_after_generate:
                self.free_memory()
                

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-1.5B", voice_to_clone=None, **kwargs):
        """Cache key for ComfyUI"""
        voice_hash = hash(str(voice_to_clone)) if voice_to_clone else 0
        return f"{hash(text)}_{model}_{voice_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"