# Created by Fabio Sarracino

import logging
import os
import re
import tempfile
import torch
import time
import numpy as np
from typing import List, Optional

from .base_vibevoice import BaseVibeVoiceNode, BufferedPyAudioStreamer
from .replace_multiple_node import ReplaceStringMultipleNode
from modular.wav2lip_streamer import Wav2LipStreamer

# Setup logging
logger = logging.getLogger("VibeVoice")

class VibeVoiceMultipleSpeakersNode(BaseVibeVoiceNode):
    def __init__(self):
        super().__init__()
        # Register this instance for memory management
        try:
            from .free_memory_node import VibeVoiceFreeMemoryNode
            VibeVoiceFreeMemoryNode.register_multi_speaker(self)
        except:
            pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "[1]: Hello, this is the first speaker.\n[2]: Hi there, I'm the second speaker.\n[1]: Nice to meet you!\n[2]: Nice to meet you too!", 
                    "tooltip": "Text with speaker labels. Use '[N]:' format where N is 1-4. Gets disabled when connected to another node.",
                    "forceInput": False,
                    "dynamicPrompts": True
                }),
                "model": (["VibeVoice-1.5B", "VibeVoice-7B", "VibeVoice-1.5B-no-llm-bf16", "VibeVoice-7B-no-llm-bf16"], {
                    "default": "VibeVoice-7B-Preview",
                    "tooltip": "Model to use. VibeVoice-7B is recommended for multi-speaker generation"
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
                "diffusion_steps": ("INT", {"default": 15, "min": 2, "max": 50, "step": 1, "tooltip": "Number of denoising steps. More steps = better quality but slower. Default: 20"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32-1, "tooltip": "Random seed for generation. Default 42 is used in official examples"}),
                "cfg_scale": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 3.0, "step": 0.05, "tooltip": "Classifier-free guidance scale (official default: 1.3)"}),
                "use_sampling": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling mode. When False (default), uses deterministic generation like official examples"}),
            },
            "optional": {
                "speaker1_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 1. If not provided, synthetic voice will be used."}),
                "speaker2_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 2. If not provided, synthetic voice will be used."}),
                "speaker3_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 3. If not provided, synthetic voice will be used."}),
                "speaker4_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 4. If not provided, synthetic voice will be used."}),
                "speaker5_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 5. If not provided, synthetic voice will be used."}),
                "speaker6_voice": ("AUDIO", {"tooltip": "Optional: Voice sample for Speaker 6. If not provided, synthetic voice will be used."}),
                "temperature": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 2.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Only used when sampling is enabled"}),
                "streaming_to_audio": ("BOOLEAN", {"default": False, "tooltip": "Enable streaming mode, playback directly to your default audio device"}),
                "streaming_buffer": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1, "tooltip": "Seconds to buffer before playing when streaming. Default: 5"}),
                "negative_llm_steps_to_skip": ("FLOAT", {"default": 0.0, "min": 0, "max": 0.9, "step": 0.1, "tooltip": "Cache and skip % of steps of negative conditioning for speed-up. 0 - cache turned off for best quality; 0.4 - optimal; 0.9 - skip 90% of steps (may cause noise)"}),
                "semantic_steps_to_skip": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.1, "tooltip": "Cache and skip % of steps of semantic encoding for speed-up. 0 - cache turned off for best quality; 1.0 - skip all except first steps, for speed"}),
                "increase_cfg": ("BOOLEAN", {"default": False, "tooltip": "Increase CFG +50% for first 50% of diffusion steps (Experimental, for more emotions)"}),
                "solver_order": (["1", "2", "3"], {
                    "default": "2",
                    "tooltip": "DPM solver_order. 2 - original. 1 - little faster, affects quality."
                }),
                "split_by": (["none", "newline", "sentence"], {
                    "default": "newline",
                    "tooltip": "Split long text into chunks by newline or by each sentence. Original: none"
                }),
                "restart_at_garbage": ("BOOLEAN", {"default": True, "tooltip": "Stop generation when garbage audio is detected"}),
                "use_compile": ("BOOLEAN", {"default": False, "tooltip": "Use torch.compile for a little speed-up. triton is needed. Speed-up is really very little."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "VibeVoiceWrapper"
    DESCRIPTION = "Generate multi-speaker conversations with up to 4 distinct voices using Microsoft VibeVoice"

    def _prepare_voice_sample(self, voice_audio, speaker_idx: int) -> Optional[np.ndarray]:
        """Prepare a single voice sample from input audio"""
        return self._prepare_audio_from_comfyui(voice_audio)
    
    def _convert_text_format(self, text: str) -> tuple:
        """Convert text with [N]: format to Speaker (N-1): format and detect speakers"""
        bracket_pattern = r'\[(\d+)\]\s*:'
        speakers_numbers = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))
        
        # Limit to 1-6 speakers
        #if not speakers_numbers:
        #    num_speakers = 1  # Default to 1 if no speaker format found
        #else:
        #    num_speakers = min(max(speakers_numbers), 6)  # Max speaker number, capped at 4
        #    if max(speakers_numbers) > 6:
        #        print(f"[VibeVoice] Warning: Found {max(speakers_numbers)} speakers, limiting to 6")
        
        # Direct conversion from [N]: to Speaker (N-1): for VibeVoice processor
        converted_text = text
        
        # Find all [N]: patterns in the text
        speakers_in_text = sorted(list(set([int(m) for m in re.findall(bracket_pattern, text)])))       
        
        if not speakers_in_text:           
            # No [N]: format found, try Speaker N: format
            speaker_pattern = r'Speaker\s+(\d+)\s*:'
            speakers_in_text = sorted(list(set([int(m) for m in re.findall(speaker_pattern, text)])))
            
            if speakers_in_text:
                # Text already in Speaker N format, convert to 0-based
                for speaker_num in sorted(speakers_in_text, reverse=True):
                    pattern = f'Speaker\\s+{speaker_num}\\s*:'
                    replacement = f'Speaker {speaker_num - 1}:'
                    converted_text = re.sub(pattern, replacement, converted_text)
            else:
                # No speaker format found
                speakers_in_text = [1]
                # Clean up newlines before assigning to speaker
                #text_clean = text.replace('\n', ' ').replace('\r', ' ')
                #text_clean = ' '.join(text_clean.split())
                converted_text = f"Speaker 0: {text}"
        else:
            # Convert [N]: directly to Speaker (N-1): and handle multi-line text
            # Split text to preserve speaker segments while cleaning up newlines within each segment
            segments = []
            
            # Find all speaker markers with their positions
            speaker_matches = list(re.finditer(f'\\[({"|".join(map(str, speakers_in_text))})\\]\\s*:', converted_text))
            
            for i, match in enumerate(speaker_matches):
                speaker_num = int(match.group(1))
                start = match.end()
                
                # Find where this speaker's text ends (at next speaker or end of text)
                if i + 1 < len(speaker_matches):
                    end = speaker_matches[i + 1].start()
                else:
                    end = len(converted_text)
                
                # Extract and clean the speaker's text
                speaker_text = converted_text[start:end].strip()
                # Replace newlines with spaces within each speaker's text
                #speaker_text = speaker_text.replace('\n', ' ').replace('\r', ' ')
                # Clean up multiple spaces
                #speaker_text = ' '.join(speaker_text.split())
                
                # Add the cleaned segment with proper speaker label
                segments.append(f'Speaker {speaker_num - 1}: {speaker_text}')
            
            # Join all segments with newlines (required for multi-speaker format)
            converted_text = '\n'.join(segments)
        
        
        speakers_in_text = [x - 1 for x in speakers_in_text] # N-1
        return converted_text, speakers_in_text
    
    def _get_speakers_in_chunk(self, chunk_text: str) -> list:
        """Extract speaker indices from a text chunk"""
        speaker_pattern = r'Speaker\s+(\d+)\s*:'
        speakers = sorted(list(set([int(m) for m in re.findall(speaker_pattern, chunk_text)])))
        return speakers
    
    def generate_speech(self, text: str = "", model: str = "VibeVoice-7B-Preview",
                       attention_type: str = "auto", free_memory_after_generate: bool = True,
                       diffusion_steps: int = 20, seed: int = 42, cfg_scale: float = 1.3,
                       use_sampling: bool = False, 
                       speaker1_voice=None, speaker2_voice=None, 
                       speaker3_voice=None, speaker4_voice=None,
                       speaker5_voice=None, speaker6_voice=None,
                       temperature: float = 0.95, top_p: float = 0.95, quantization_mode = "bf16", 
                       streaming_to_audio = False, streaming_to_wav2lip = False, streaming_buffer = 10,
                       exllama_model = "None", 
                       negative_llm_steps_to_skip: float = 0.0, 
                       semantic_steps_to_skip: float = 0.0, 
                       increase_cfg: bool=False,
                       solver_order: int = 2,
                       split_by: str="newline", 
                       restart_at_garbage: bool=False, use_compile: bool=False):
        """Generate multi-speaker speech from text using VibeVoice"""
        
        audio_dict = None
        
        try:
            # Check text input
            if not text or not text.strip():
                raise Exception("No text provided. Please enter text with speaker labels (e.g., '[1]: Hello' or '[2]: Hi')")
            
            streaming = streaming_to_audio
            
            # Get model mapping and load model with attention type
            model_mapping = self._get_model_mapping()
            model_path = model_mapping.get(model, model)
            self.load_model(model_path, attention_type, quantization_mode, exllama_model)
            
            # Prepare voice inputs
            voice_inputs = [speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice, speaker5_voice, speaker6_voice]
            
            # Convert text format and detect speakers
            converted_text, speakers_in_text = self._convert_text_format(text)
            
            # Prepare all voice samples (for all speakers in the entire text)
            all_voice_samples = [None] * 6
            for i, speaker_num in enumerate(speakers_in_text):
                idx = speaker_num  # Converted to 0-based for voice array
                
                # Try to use provided voice sample
                if idx < len(voice_inputs) and voice_inputs[idx] is not None:
                    voice_sample = self._prepare_voice_sample(voice_inputs[idx], idx)
                    if voice_sample is None:                      
                        # Use the actual speaker index for consistent synthetic voice
                        voice_sample = self._create_synthetic_voice_sample(idx)
                else:
                    # Use the actual speaker index for consistent synthetic voice
                    voice_sample = self._create_synthetic_voice_sample(idx)                    
                    
                all_voice_samples[idx] = voice_sample
            
            # Ensure voice_samples count matches detected speakers
            # Count how many items are not None
            non_none_count_all_voice_samples = sum(1 for item in all_voice_samples if item is not None)

            if non_none_count_all_voice_samples != len(speakers_in_text):
                logger.error(f"Mismatch: {len(speakers_in_text)} speakers but {non_none_count_all_voice_samples} voice samples!")
                raise Exception(f"Voice sample count mismatch: expected {len(speakers_in_text)}, got {non_none_count_all_voice_samples}")
            
            max_new_tokens = 32637
            sample_rate = 24000
            
            # Split text into chunks if enabled           
            text_chunks = [converted_text.strip()]
            if split_by == "newline":
                chunks = [chunk.strip() for chunk in converted_text.split('\n') if chunk.strip()]
                if chunks:
                    text_chunks = chunks
            elif split_by == "sentence": 
                chunks = self.split_by_sentence_enhanced(converted_text)
                if chunks:
                    text_chunks = chunks
            
            speaker_prev = "0"
            if streaming or streaming_to_wav2lip:
                # Streaming mode with chunk processing
                audio_streamer = BufferedPyAudioStreamer(sample_rate=sample_rate, buffer_duration=streaming_buffer, restart_at_garbage=restart_at_garbage)
                audio_streamer.start_playback()

                # format "Speaker 1: some text"
                for chunk_text in text_chunks:
                    if chunk_text.startswith("Speaker "):
                        speaker_prev = chunk_text[8:9]
                    else:
                        chunk_text = "Speaker "+speaker_prev+": " + chunk_text
                    
                    if chunk_text.endswith(',') or chunk_text.endswith(':'):
                        chunk_text = chunk_text[:-1] + '.' # was a trailing coma
                    # Add dot at the end if no proper punctuation exists
                    if chunk_text and not chunk_text.endswith(('.', '!', '?')):
                        chunk_text += '.'
            
                    # Reset the streamer's finished flag before processing the next chunk
                    if hasattr(audio_streamer, 'finished_flags'):
                        audio_streamer.finished_flags[0] = False
                    
                    # Get speakers present in this chunk
                    chunk_speakers = self._get_speakers_in_chunk(chunk_text)
                    
                    # Prepare voice samples for only the speakers in this chunk
                    chunk_voice_samples = []
                    for speaker_idx in chunk_speakers:
                        if speaker_idx <= max(speakers_in_text):
                            chunk_voice_samples.append(all_voice_samples[speaker_idx])
                        else:                           
                            # Fallback: use first voice sample if speaker index is out of range
                            chunk_voice_samples.append(all_voice_samples[0])
                    
                    print(chunk_text)
                    if split_by == "newline" or split_by == "sentence":
                        chunk_text = chunk_text.replace("Speaker 1", "Speaker 0").replace("Speaker 2", "Speaker 0").replace("Speaker 3", "Speaker 0").replace("Speaker 4", "Speaker 0").replace("Speaker 5", "Speaker 0")
                    self._generate_with_vibevoice(
                        chunk_text, 
                        chunk_voice_samples, cfg_scale, seed, diffusion_steps, 
                        use_sampling, temperature, top_p, 
                        audio_streamer=audio_streamer,
                        max_new_tokens=max_new_tokens,
                        negative_llm_steps_to_skip=negative_llm_steps_to_skip,
                        semantic_steps_to_skip=semantic_steps_to_skip,
                        increase_cfg=increase_cfg,
                        restart_at_garbage=restart_at_garbage,
                        use_compile=use_compile
                    )

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
                # Non-streaming mode
                all_audio_segments = []
                for i, chunk_text in enumerate(text_chunks):
                    if chunk_text.startswith("Speaker "):
                        speaker_prev = chunk_text[8:9]
                    else:
                        chunk_text = "Speaker "+speaker_prev+": " + chunk_text
                        
                    if chunk_text.endswith(',') or chunk_text.endswith(':'):
                        chunk_text = chunk_text[:-1] + '.' # was a trailing coma
                    # Add dot at the end if no proper punctuation exists
                    if chunk_text and not chunk_text.endswith(('.', '!', '?')):
                        chunk_text += '.'
                    # Get speakers present in this chunk
                    chunk_speakers = self._get_speakers_in_chunk(chunk_text)
                    
                    # Prepare voice samples for only the speakers in this chunk
                    chunk_voice_samples = []
                    for speaker_idx in chunk_speakers:
                        if speaker_idx < len(all_voice_samples):
                            chunk_voice_samples.append(all_voice_samples[speaker_idx])
                        else:
                            # Fallback: use first voice sample if speaker index is out of range
                            chunk_voice_samples.append(all_voice_samples[0])
                    
                    if split_by == "newline" or split_by == "sentence":
                        chunk_text = chunk_text.replace("Speaker 1", "Speaker 0").replace("Speaker 2", "Speaker 0").replace("Speaker 3", "Speaker 0").replace("Speaker 4", "Speaker 0").replace("Speaker 5", "Speaker 0")
                    print(chunk_text)
                    chunk_audio_dict = self._generate_with_vibevoice(
                        chunk_text, 
                        chunk_voice_samples, cfg_scale, seed, diffusion_steps, 
                        use_sampling, temperature, top_p, 
                        streaming=streaming, buffer_duration=streaming_buffer,
                        max_new_tokens=max_new_tokens, 
                        negative_llm_steps_to_skip=negative_llm_steps_to_skip,
                        semantic_steps_to_skip=semantic_steps_to_skip,
                        increase_cfg=increase_cfg,
                        restart_at_garbage=restart_at_garbage,
                        use_compile=use_compile
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
            
            # Free memory if requested
            if free_memory_after_generate:
                self.free_memory()
            
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
                

    @classmethod
    def IS_CHANGED(cls, text="", model="VibeVoice-7B-Preview",
                   speaker1_voice=None, speaker2_voice=None, 
                   speaker3_voice=None, speaker4_voice=None, **kwargs):
        """Cache key for ComfyUI"""
        voices_hash = hash(str([speaker1_voice, speaker2_voice, speaker3_voice, speaker4_voice]))
        return f"{hash(text)}_{model}_{voices_hash}_{kwargs.get('cfg_scale', 1.3)}_{kwargs.get('seed', 0)}"