# Created by Fabio Sarracino
# Base class for VibeVoice nodes with common functionality

import logging
import os
import tempfile
import torch
import numpy as np
import re
from typing import List, Optional, Tuple, Any
import threading
import pyaudio
from queue import Queue
import time

# Setup logging
logger = logging.getLogger("VibeVoice")

# Import for interruption support
try:
    import execution
    INTERRUPTION_SUPPORT = True
except ImportError:
    INTERRUPTION_SUPPORT = False
    logger.warning("Interruption support not available")

# Check for SageAttention availability
try:
    from sageattention import sageattn
    SAGE_AVAILABLE = True
    logger.info("SageAttention available for acceleration")
except ImportError:
    SAGE_AVAILABLE = False
    logger.debug("SageAttention not available - install with: pip install sageattention")

def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
        
class BufferedPyAudioStreamer:
    """PyAudio-based streamer with buffer accumulation and full audio collection"""
    def __init__(self, sample_rate=24000, buffer_duration=10.0):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_samples = int(sample_rate * buffer_duration)
        self.audio_queue = Queue()
        # This flag is likely checked by the model, so we must manage it.
        self.finished_flags = [False] 
        self.playing = False
        self.audio_thread = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio = np.array([], dtype=np.float32)

    def put(self, audio_chunk, indices):
        """Add audio chunk to the buffer and full audio collection"""
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.cpu().float().numpy()
        
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
            
        self.full_audio = np.concatenate([self.full_audio, audio_chunk]) if self.full_audio.size else audio_chunk
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk]) if self.audio_buffer.size else audio_chunk
        
        if len(self.audio_buffer) >= self.buffer_samples:
            play_chunk = self.audio_buffer[:self.buffer_samples]
            self.audio_buffer = self.audio_buffer[self.buffer_samples:]
            self.audio_queue.put(play_chunk)

    # --- CHANGED: This is now a "soft end" for flushing the buffer ---
    def end(self, indices=None):
        """
        Called by model.generate() after each chunk.
        Flushes the remaining audio buffer for the chunk but does NOT terminate the playback thread.
        """
        self.finished_flags[0] = True # Signal to the model this chunk is done.
        
        # If there's anything left in the buffer, queue it for playing.
        if self.audio_buffer.size > 0:
            play_chunk = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32) # Clear the buffer
            self.audio_queue.put(play_chunk)
        # CRITICAL: We no longer put `None` in the queue here.

    # --- NEW: This is the "hard end" to terminate the thread ---
    def close(self):
        """
        Called once all chunks are processed.
        Sends the termination signal (None) to the playback thread.
        """
        self.audio_queue.put(None)

    def get_full_audio(self):
        """Return the complete audio that was generated"""
        return self.full_audio

    def _audio_playback_thread(self):
        """Thread function for audio playback (no changes needed here)"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, output=True)
        self.playing = True
        try:
            while self.playing:
                chunk = self.audio_queue.get()
                if chunk is None:
                    break # The thread will exit cleanly when it receives None
                stream.write(chunk.astype(np.float32).tobytes())
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.playing = False

    def start_playback(self):
        """Start the audio playback thread (no changes needed here)"""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.audio_thread = threading.Thread(target=self._audio_playback_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            
class BaseVibeVoiceNode:
    """Base class for VibeVoice nodes containing common functionality"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_path = None
        self.current_attention_type = None
        self.current_quantization_mode = None # NEW: Track current quantization mode
        
    def free_memory(self):
        """Free model and processor from memory"""
        try:
            if self.model.exllama is not None:
                self.model.exllama.unload()
                self.model.exllama = None    
            
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
                
            
            self.current_model_path = None
            self.current_quantization_mode = None # NEW: Reset quantization mode                                                                                
            
            # Force garbage collection and clear CUDA cache if available
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Model and processor memory freed successfully")
            
        except Exception as e:
            logger.error(f"Error freeing memory: {e}")
    
    def _check_dependencies(self):
        """Check if VibeVoice is available and import it with fallback installation"""
        try:
            import sys
            import os
            import shutil
            
            # Add vvembed to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            vvembed_path = os.path.join(parent_dir, 'vvembed')
            
            # Remove any existing vibevoice modules to force re-import from new path
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('vibevoice'):
                    del sys.modules[module_name]
            
            # Insert the embedded path at the beginning of sys.path
            if vvembed_path not in sys.path:
                sys.path.insert(0, vvembed_path)
            
            print(f"Using embedded VibeVoice from {vvembed_path}")
            
            # Import from embedded version
            from modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference, 
                ExLlamaV3Wrapper
            )
            return None, VibeVoiceForConditionalGenerationInference, ExLlamaV3Wrapper
            
        except ImportError as e:
            print(f"Embedded VibeVoice import failed: {e}")
            
            # Try fallback to installed version if available
            try:
                import vibevoice
                from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
                print("Falling back to system-installed VibeVoice")
                return vibevoice, VibeVoiceForConditionalGenerationInference
            except ImportError:
                pass
            
            raise Exception(
                "VibeVoice embedded module import failed. Please ensure the vvembed folder exists "
                "and transformers>=4.51.3 is installed."
            )            
    
    def _apply_sage_attention(self):
        """Apply SageAttention to the loaded model by monkey-patching attention layers"""
        try:
            from sageattention import sageattn
            import torch.nn.functional as F
            
            # Counter for patched layers
            patched_count = 0
            
            def patch_attention_forward(module):
                """Recursively patch attention layers to use SageAttention"""
                nonlocal patched_count
                
                # Check if this module has scaled_dot_product_attention
                if hasattr(module, 'forward'):
                    original_forward = module.forward
                    
                    # Create wrapper that replaces F.scaled_dot_product_attention with sageattn
                    def sage_forward(*args, **kwargs):
                        # Temporarily replace F.scaled_dot_product_attention
                        original_sdpa = F.scaled_dot_product_attention
                        
                        def sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
                            """Wrapper that converts sdpa calls to sageattn"""
                            # Log any unexpected parameters for debugging
                            if kwargs:
                                unexpected_params = list(kwargs.keys())
                                logger.debug(f"SageAttention: Ignoring unsupported parameters: {unexpected_params}")
                            
                            try:
                                # SageAttention expects tensors in specific format
                                # Transformers typically use (batch, heads, seq_len, head_dim)
                                
                                # Check tensor dimensions to determine layout
                                if query.dim() == 4:
                                    # 4D tensor: (batch, heads, seq, dim)
                                    batch_size = query.shape[0]
                                    num_heads = query.shape[1]
                                    seq_len_q = query.shape[2]
                                    seq_len_k = key.shape[2]
                                    head_dim = query.shape[3]
                                    
                                    # Reshape to (batch*heads, seq, dim) for HND layout
                                    query_reshaped = query.reshape(batch_size * num_heads, seq_len_q, head_dim)
                                    key_reshaped = key.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    value_reshaped = value.reshape(batch_size * num_heads, seq_len_k, head_dim)
                                    
                                    # Call sageattn with HND layout
                                    output = sageattn(
                                        query_reshaped, key_reshaped, value_reshaped,
                                        is_causal=is_causal,
                                        tensor_layout="HND"  # Heads*batch, seqN, Dim
                                    )
                                    
                                    # Output should be (batch*heads, seq_len_q, head_dim)
                                    # Reshape back to (batch, heads, seq, dim)
                                    if output.dim() == 3:
                                        output = output.reshape(batch_size, num_heads, seq_len_q, head_dim)
                                    
                                    return output
                                else:
                                    # For 3D tensors, assume they're already in HND format
                                    output = sageattn(
                                        query, key, value,
                                        is_causal=is_causal,
                                        tensor_layout="HND"
                                    )
                                    return output
                                    
                            except Exception as e:
                                # If SageAttention fails, fall back to original implementation
                                logger.debug(f"SageAttention failed, using original: {e}")
                                # Call with proper arguments - scale is a keyword argument in PyTorch 2.0+
                                # Pass through any additional kwargs that the original sdpa might support
                                if scale is not None:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
                                else:
                                    return original_sdpa(query, key, value, attn_mask=attn_mask, 
                                                       dropout_p=dropout_p, is_causal=is_causal, **kwargs)
                        
                        # Replace the function
                        F.scaled_dot_product_attention = sage_sdpa
                        
                        try:
                            # Call original forward with patched attention
                            result = original_forward(*args, **kwargs)
                        finally:
                            # Restore original function
                            F.scaled_dot_product_attention = original_sdpa
                        
                        return result
                    
                    # Check if this module likely uses attention
                    # Look for common attention module names
                    module_name = module.__class__.__name__.lower()
                    if any(name in module_name for name in ['attention', 'attn', 'multihead']):
                        module.forward = sage_forward
                        patched_count += 1
                
                # Recursively patch child modules
                for child in module.children():
                    patch_attention_forward(child)
            
            # Apply patching to the entire model
            patch_attention_forward(self.model)
            
            logger.info(f"Patched {patched_count} attention layers with SageAttention")
            
            if patched_count == 0:
                logger.warning("No attention layers found to patch - SageAttention may not be applied")
                
        except Exception as e:
            logger.error(f"Failed to apply SageAttention: {e}")
            logger.warning("Continuing with standard attention implementation")                                                           
    
    def _load_exllama_model(self, exllama_model_handle: str, comfyui_models_dir: str):
        """Load ExLlama model with download if not found locally"""
        try:
            mapping = self._get_model_mapping()
            if exllama_model_handle not in mapping:
                raise ValueError(f"ExLlama model handle {exllama_model_handle} not found in model mapping")

            model_path = mapping[exllama_model_handle]
            # The local directory in ComfyUI for this model
            model_dir = os.path.join(comfyui_models_dir, f"models--{model_path.replace('/', '--')}")

            # Check if the model exists locally, else download
            if not os.path.exists(model_dir):
                from huggingface_hub import snapshot_download
                logger.info(f"Downloading ExLlama model {model_path} to {model_dir}")
                model_dir = snapshot_download(repo_id=model_path, cache_dir=comfyui_models_dir)

            # Determine the actual model directory (handle both old and new structures)
            actual_model_dir = model_dir
            if not os.path.exists(os.path.join(model_dir, "model.safetensors")):
                # Check for snapshots directory
                snapshots_dir = os.path.join(model_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    # Get all snapshot subdirectories
                    snapshots = [d for d in os.listdir(snapshots_dir) 
                               if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        # Use the first snapshot directory (typically only one exists)
                        actual_model_dir = os.path.join(snapshots_dir, snapshots[0])
                    else:
                        raise FileNotFoundError("No snapshot directories found in {snapshots_dir}")
                else:
                    raise FileNotFoundError("model.safetensors not found in root and no snapshots directory")

            # Now load the ExLlama model from actual_model_dir
            from exllamav3 import Config, Model, Cache
            exllama_config = Config.from_directory(actual_model_dir)
            exllama_model = Model.from_config(exllama_config)
            exllama_positive_cache = Cache(exllama_model, max_num_tokens=2048)
            exllama_negative_cache = Cache(exllama_model, max_num_tokens=2048)
            exllama_model.load()

            from modular.modeling_vibevoice_inference import ExLlamaV3Wrapper
            return ExLlamaV3Wrapper(
                model=exllama_model,
                positive_cache=exllama_positive_cache,
                negative_cache=exllama_negative_cache,
                config=exllama_config
            )

        except Exception as e:
            logger.error(f"Failed to load ExLlama model: {e}")
            raise
        
    
    def load_model(self, model_path: str, attention_type: str = "auto", quantization_mode: str = "bf16", exllama_model: str = "None" ):
        """Load VibeVoice model with specified attention implementation and quantization"""        
        
        # NEW: Update the check to include quantization_mode
        if (self.model is None or 
            getattr(self, 'current_model_path', None) != model_path or
            getattr(self, 'current_attention_type', None) != attention_type or
            getattr(self, 'current_quantization_mode', None) != quantization_mode or
            getattr(self, 'current_exllama_model', None) != exllama_model):
            try:
                vibevoice, VibeVoiceInferenceModel, ExLlamaV3Wrapper = self._check_dependencies()
                
                self.free_memory()
                
                # Set ComfyUI models directory
                import folder_paths
                models_dir = folder_paths.get_folder_paths("checkpoints")[0]
                comfyui_models_dir = os.path.join(os.path.dirname(models_dir), "vibevoice")
                os.makedirs(comfyui_models_dir, exist_ok=True)
                
                # Force HuggingFace to use ComfyUI directory
                original_hf_home = os.environ.get('HF_HOME')
                original_hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
                
                os.environ['HF_HOME'] = comfyui_models_dir
                os.environ['HUGGINGFACE_HUB_CACHE'] = comfyui_models_dir
                use_exllama = False
                
                # Import time for timing
                import time
                start_time = time.time()
                
                # Suppress verbose logs
                import transformers
                import warnings
                transformers.logging.set_verbosity_error()
                warnings.filterwarnings("ignore", category=UserWarning)
                
                # Check if model exists locally
                model_dir = os.path.join(comfyui_models_dir, f"models--{model_path.replace('/', '--')}")
                model_exists_in_comfyui = os.path.exists(model_dir)
                
                # Prepare attention implementation kwargs
                model_kwargs = {
                    "cache_dir": comfyui_models_dir,
                    "trust_remote_code": True,                    
                    "device_map": "cuda" if torch.cuda.is_available() else "cpu",
                }
                
                if exllama_model != "None":
                    use_exllama = True
                    
                config = None  # Initialize config variable    
                
                if use_exllama:
                    print(f"exllama model {exllama_model} is loading...")
                    exllama_wrapper = self._load_exllama_model(exllama_model, comfyui_models_dir)
                
                # NEW: Configure quantization based on the selected mode
                if quantization_mode == "bnb_nf4":
                    logger.info("Loading model with bnb_nf4 quantization to reduce VRAM.")
                    try:
                        from transformers import BitsAndBytesConfig
                    except ImportError:
                        logger.error("bitsandbytes is not installed. Please install it with: pip install bitsandbytes")
                        raise ImportError("The 'bitsandbytes' library is required for bnb_nf4 quantization.")

                    if not torch.cuda.is_available():
                        raise ValueError("bnb_nf4 quantization requires a CUDA-enabled GPU.")

                    # Define the 4-bit quantization configuration
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bf16 for speed and precision
                        bnb_4bit_use_double_quant=True,
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    # torch_dtype is handled by the compute_dtype in BitsAndBytesConfig
                    
                elif quantization_mode == "bnb_8bit":
                    logger.info("Loading model with bnb_8bit quantization.")
                    try:
                        # Ensure bitsandbytes is available
                        import bitsandbytes
                    except ImportError:
                        logger.error("bitsandbytes is not installed. Please install it with: pip install bitsandbytes")
                        raise ImportError("The 'bitsandbytes' library is required for bnb_8bit quantization.")

                    if not torch.cuda.is_available():
                        raise ValueError("bnb_8bit quantization requires a CUDA-enabled GPU.")                    
                    model_kwargs["load_in_8bit"] = True
                    
                elif quantization_mode == "bf16":
                    logger.info("Loading model in bfloat16 (original behavior).")
                    model_kwargs["torch_dtype"] = torch.bfloat16
                    
                elif quantization_mode == "fp16":
                    logger.info("Loading model in float16.")
                    model_kwargs["torch_dtype"] = torch.float16     
                    
                elif quantization_mode == "float8_e4m3fn":
                    logger.info("Loading model in float8_e4m3fn")
                    model_kwargs["torch_dtype"] = torch.float8_e4m3fn
                
                else: # Default to float32 if an unknown mode is given
                    logger.info("Loading model in float32.")

                # Set attention implementation based on user selection
                use_sage_attention = False
                if attention_type == "sage":
                    # SageAttention requires special handling - can't be set via attn_implementation
                    if not SAGE_AVAILABLE:
                        logger.warning("SageAttention not installed, falling back to sdpa")
                        logger.warning("Install with: pip install sageattention")
                        model_kwargs["attn_implementation"] = "sdpa"
                    elif not torch.cuda.is_available():
                        logger.warning("SageAttention requires CUDA GPU, falling back to sdpa")
                        model_kwargs["attn_implementation"] = "sdpa"
                    else:
                        # Don't set attn_implementation for sage, will apply after loading
                        use_sage_attention = True
                        logger.info("Will apply SageAttention after model loading")
                elif attention_type != "auto":
                    model_kwargs["attn_implementation"] = attention_type
                    logger.info(f"Using {attention_type} attention implementation")
                else:
                    # Auto mode - let transformers decide the best implementation
                    logger.info("Using auto attention implementation selection")
                
                # Try to load locally first
                try:                        
                    if model_exists_in_comfyui:
                        model_kwargs["local_files_only"] = True
                        self.model = VibeVoiceInferenceModel.from_pretrained(
                            model_path,
                            **model_kwargs
                        )
                    else:
                        raise FileNotFoundError("Model not found locally")
                except (FileNotFoundError, OSError) as e:
                    logger.info(f"Downloading {model_path}...")
                    model_kwargs["local_files_only"] = False
                    self.model = VibeVoiceInferenceModel.from_pretrained(
                        model_path,
                        **model_kwargs
                    )
                    elapsed = time.time() - start_time
                else:
                    elapsed = time.time() - start_time
                    
                # Assign the pre-loaded ExLlama wrapper to the HF model
                if use_exllama:
                    self.model.exllama = exllama_wrapper               
                
                # Load processor
                from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
                self.processor = VibeVoiceProcessor.from_pretrained(model_path)
                
                # Restore environment variables
                if original_hf_home is not None:
                    os.environ['HF_HOME'] = original_hf_home
                elif 'HF_HOME' in os.environ:
                    del os.environ['HF_HOME']
                    
                if original_hf_cache is not None:
                    os.environ['HUGGINGFACE_HUB_CACHE'] = original_hf_cache
                elif 'HUGGINGFACE_HUB_CACHE' in os.environ:
                    del os.environ['HUGGINGFACE_HUB_CACHE']
                
                # Apply SageAttention if requested and available
                if use_sage_attention and SAGE_AVAILABLE:
                    self._apply_sage_attention()
                    logger.info("SageAttention successfully applied to model")
                    
                self.current_model_path = model_path
                self.current_attention_type = attention_type
                self.current_quantization_mode = quantization_mode # NEW: Save the current mode
                self.current_exllama_model = exllama_model # NEW: Save the current mode
            except Exception as e:
                logger.error(f"Failed to load VibeVoice model: {str(e)}")
                if 'exllama_wrapper' in locals():
                    exllama_wrapper.unload()
                    exllama_wrapper = None    
                    print("deleting exllama_wrapper")
                    del exllama_wrapper
                self.free_memory()
                raise Exception(f"Base model loading failed: {str(e)}")
    
    def _create_synthetic_voice_sample(self, speaker_idx: int) -> np.ndarray:
        """Create synthetic voice sample for a specific speaker"""
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        t = np.linspace(0, duration, samples, False)
        
        # Create realistic voice-like characteristics for each speaker
        # Use different base frequencies for different speaker types
        base_frequencies = [120, 180, 140, 200]  # Mix of male/female-like frequencies
        base_freq = base_frequencies[speaker_idx % len(base_frequencies)]
        
        # Create vowel-like formants (like "ah" sound) - unique per speaker
        formant1 = 800 + speaker_idx * 100  # First formant
        formant2 = 1200 + speaker_idx * 150  # Second formant
        
        # Generate more voice-like waveform
        voice_sample = (
            # Fundamental with harmonics (voice-like)
            0.6 * np.sin(2 * np.pi * base_freq * t) +
            0.25 * np.sin(2 * np.pi * base_freq * 2 * t) +
            0.15 * np.sin(2 * np.pi * base_freq * 3 * t) +
            
            # Formant resonances (vowel-like characteristics)
            0.1 * np.sin(2 * np.pi * formant1 * t) * np.exp(-t * 2) +
            0.05 * np.sin(2 * np.pi * formant2 * t) * np.exp(-t * 3) +
            
            # Natural breath noise (reduced)
            0.02 * np.random.normal(0, 1, len(t))
        )
        
        # Add natural envelope (like human speech pattern)
        # Quick attack, slower decay with slight vibrato (unique per speaker)
        vibrato_freq = 4 + speaker_idx * 0.3  # Slightly different vibrato per speaker
        envelope = (np.exp(-t * 0.3) * (1 + 0.1 * np.sin(2 * np.pi * vibrato_freq * t)))
        voice_sample *= envelope * 0.08  # Lower volume
        
        return voice_sample.astype(np.float32)
    
    def _prepare_audio_from_comfyui(self, voice_audio, target_sample_rate: int = 24000) -> Optional[np.ndarray]:
        """Prepare audio from ComfyUI format to numpy array"""
        if voice_audio is None:
            return None
            
        # Extract waveform from ComfyUI audio format
        if isinstance(voice_audio, dict) and "waveform" in voice_audio:
            waveform = voice_audio["waveform"]
            input_sample_rate = voice_audio.get("sample_rate", target_sample_rate)
            
            # Convert to numpy
            if isinstance(waveform, torch.Tensor):
                audio_np = waveform.cpu().numpy()
            else:
                audio_np = np.array(waveform)
            
            # Handle different audio shapes
            if audio_np.ndim == 3:  # (batch, channels, samples)
                audio_np = audio_np[0, 0, :]  # Take first batch, first channel
            elif audio_np.ndim == 2:  # (channels, samples)
                audio_np = audio_np[0, :]  # Take first channel
            # If 1D, leave as is
            
            # Resample if needed
            if input_sample_rate != target_sample_rate:
                target_length = int(len(audio_np) * target_sample_rate / input_sample_rate)
                audio_np = np.interp(np.linspace(0, len(audio_np), target_length), 
                                   np.arange(len(audio_np)), audio_np)
            
            # Ensure audio is in correct range [-1, 1]
            audio_max = np.abs(audio_np).max()
            if audio_max > 0:
                audio_np = audio_np / max(audio_max, 1.0)  # Normalize
            
            return audio_np.astype(np.float32)
        
        return None
    
    def _get_model_mapping(self) -> dict:
        """Get model name mappings"""
        return {
            "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
            "VibeVoice-7B": "aoi-ot/VibeVoice-Large",
            "VibeVoice-1.5B-no-llm-bf16": "tensorbanana/vibevoice-1.5b-no-llm-bf16",
            "VibeVoice-7B-no-llm-bf16": "tensorbanana/vibevoice-7b-no-llm-bf16",
            "vibevoice-1.5b-exl3-8bit": "tensorbanana/vibevoice-1.5b-exl3-8bit",
            "vibevoice-7b-exl3-8bit": "tensorbanana/vibevoice-7b-exl3-8bit",
            "vibevoice-7b-exl3-6bit": "tensorbanana/vibevoice-7b-exl3-6bit",
            "vibevoice-7b-exl3-4bit": "tensorbanana/vibevoice-7b-exl3-4bit",
            "vibevoice-7b-exl3-3bit": "tensorbanana/vibevoice-7b-exl3-3bit",
        }
    
    def _format_text_for_vibevoice(self, text: str, speakers: list) -> str:
        """Format text with speaker information for VibeVoice using correct format"""
        # Remove any newlines from the text to prevent parsing issues
        # The processor splits by newline and expects each line to have "Speaker N:" format
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        # VibeVoice expects format: "Speaker 1: text" not "Name: text"
        if len(speakers) == 1:
            return f"Speaker 1: {text}"
        else:
            # Check if text already has proper Speaker N: format
            if re.match(r'^\s*Speaker\s+\d+\s*:', text, re.IGNORECASE):
                return text
            # If text has name format, convert to Speaker N format
            elif any(f"{speaker}:" in text for speaker in speakers):
                formatted_text = text
                for i, speaker in enumerate(speakers):
                    formatted_text = formatted_text.replace(f"{speaker}:", f"Speaker {i+1}:")
                return formatted_text
            else:
                # Plain text, assign to first speaker
                return f"Speaker 1: {text}"
    
    def _generate_with_vibevoice(self, formatted_text: str, voice_samples: List[np.ndarray], 
                             cfg_scale: float, seed: int, diffusion_steps: int, use_sampling: bool,
                             temperature: float = 0.95, top_p: float = 0.95, 
                             streaming: bool = False, buffer_duration: float = 10.0, 
                             max_new_tokens = 32637, negative_llm_steps_to_cache = 0, 
                             increase_cfg=False,
                             audio_streamer: BufferedPyAudioStreamer = None) -> dict:
        """Generate audio using VibeVoice model with optional streaming and returning full audio"""
        try:
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            self.model.negative_llm_steps_to_cache = negative_llm_steps_to_cache
            self.model.increase_cfg = increase_cfg
            self.model.set_ddpm_inference_steps(diffusion_steps)
            
            local_streamer = audio_streamer
            is_streaming_active = (local_streamer is not None) or streaming
            if local_streamer is None and streaming:
                local_streamer = BufferedPyAudioStreamer(sample_rate=24000, buffer_duration=buffer_duration)
                local_streamer.start_playback()

            inputs = self.processor([formatted_text], voice_samples=[voice_samples], return_tensors="pt", return_attention_mask=True)
            device = next(self.model.parameters()).device
            inputs = {
                k: v.to(device) if hasattr(v, 'to') else v 
                for k, v in inputs.items()
            }

            stop_check_fn = None
            if INTERRUPTION_SUPPORT:
                # --- THIS IS THE CRITICAL CHANGE ---
                # This function will now throw the exception instead of catching it and returning True.
                # This allows the exception to propagate up and stop the main chunk processing loop.
                def comfyui_stop_check():
                    import comfy.model_management as mm
                    # This line will now raise InterruptProcessingException if the user clicks stop.
                    mm.throw_exception_if_processing_interrupted()
                    # The function no longer needs to return anything, as the exception will halt execution.
                    # However, your inference loop checks the return value, so we return False if not interrupted.
                    return False 
            
                # We need to wrap the actual function call in your inference loop logic.
                # So we create a wrapper that returns True on exception for the break,
                # but more importantly, we let the original exception propagate.
                # A better way is to modify the inference loop, but let's change the check function instead.
                
                # Re-thinking: The simplest change is to NOT catch the exception.
                # Your inference loop is not shown, but it likely calls this function.
                # The key is to let the exception happen. Let's simplify the check function entirely.
                
                def final_stop_check():
                    import comfy.model_management as mm
                    # The call to model.generate is inside a try/except block.
                    # When this throws, it will be caught and propagated correctly.
                    # The `if stop_check_fn()` in your inference loop will never complete if an exception is thrown.
                    mm.throw_exception_if_processing_interrupted()

                stop_check_fn = final_stop_check

            with torch.no_grad():
                gen_kwargs = {
                    "tokenizer": self.processor.tokenizer,
                    "cfg_scale": cfg_scale,
                    "max_new_tokens": max_new_tokens,
                    "audio_streamer": local_streamer,
                    "return_speech": not is_streaming_active,
                    # Pass the function directly. The model's generate loop must call it.
                    "stop_check_fn": stop_check_fn,
                }
                if use_sampling:
                    gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
                else:
                    gen_kwargs["do_sample"] = False
                
                # The model.generate call is what needs to be interrupted.
                output = self.model.generate(**inputs, **gen_kwargs)
                
            if is_streaming_active and local_streamer:
                if audio_streamer is None:
                    local_streamer.end()
                    while local_streamer.playing:
                        time.sleep(0.1)
                    full_audio_np = local_streamer.get_full_audio()
                    waveform = torch.from_numpy(full_audio_np).unsqueeze(0).unsqueeze(0) if full_audio_np.size > 0 else None
                else:
                    return {"waveform": None, "sample_rate": 24000, "streamed": True}

                return {"waveform": waveform, "sample_rate": 24000, "streamed": True}
            else:
                speech_tensors = getattr(output, 'speech_outputs', None)
                if not speech_tensors:
                    raise Exception("VibeVoice failed to generate audio.")

                audio_tensor = speech_tensors[0] if isinstance(speech_tensors, list) else speech_tensors
                if audio_tensor.dtype == torch.bfloat16:
                    audio_tensor = audio_tensor.to(torch.float32)

                return {"waveform": audio_tensor.cpu().unsqueeze(0), "sample_rate": 24000, "streamed": False}
                    
        except Exception as e:
            # This block will now catch InterruptProcessingException from the generate call.
            logger.error(f"VibeVoice generation failed: {e}")
            if 'local_streamer' in locals() and local_streamer and audio_streamer is None:
                local_streamer.end()
            # Re-raise the exception so the main `generate_speech` function can catch it and stop.
            raise       