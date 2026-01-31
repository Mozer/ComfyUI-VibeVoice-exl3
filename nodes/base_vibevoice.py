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
from modular.wav2lip_streamer import Wav2LipStreamer

from schedule.dpm_solver import DPMSolverMultistepScheduler

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


def estimate_word_trim_position(text: str, valid_steps: int, steps_per_word: float = 4.0) -> int:
            """
            Estimates the word count that corresponds to the number of valid steps.
            Finds the nearest whitespace to avoid cutting words.
            """
            # 1. Estimate words spoken: steps / ratio
            estimated_words = int(valid_steps / steps_per_word)
            
            # 2. Split the text (excluding "Speaker 0: " prefix for now)
            # The text will be prefixed with 'Speaker 0: ' in the loop, so account for it.
            prefix_len = len("Speaker 0: ")
            
            # Strip the speaker prefix if present (only do this for the original text)
            if text.startswith("Speaker 0: "):
                clean_text = text[prefix_len:].strip()
            else:
                clean_text = text.strip()

            words = clean_text.split()
            
            # 3. Safety check: Don't trim more words than exist
            trim_word_count = min(estimated_words, len(words))
            
            if trim_word_count <= 0:
                return 0 # Trim nothing
            
            # 4. Reconstruct the trimmed part and find its index in the original text
            trimmed_words = words[:trim_word_count]
            trimmed_text_part = " ".join(trimmed_words)
            
            # Find the index of the character *after* the trimmed part in the clean text.
            # We add 1 for the space separator, if it exists.
            trim_char_index_in_clean_text = len(trimmed_text_part)
            
            # Look for the next space to ensure we trim a full word.
            # We search in the clean text starting from the estimated position.
            next_space = clean_text.find(' ', trim_char_index_in_clean_text)
            
            if next_space != -1:
                # Trim at the space *after* the last spoken word
                trim_char_index_in_clean_text = next_space
            else:
                # If no space found ahead, it means the last word was the final word.
                # In this case, we trim to the end of the last word.
                pass 
                
            # The final index must be in the original formatted_text
            # The logic above assumes the text has been pre-cleaned. We need the index *in the original string*
            # For simplicity and robustness, we find the index of the first character of the word *after* the cut.
            
            # If we trim 0 words, the index is 0.
            if trim_word_count == 0:
                 return 0
                 
            # Find the starting position of the next word to be spoken
            # This is the character index in the *clean* text
            if trim_word_count < len(words):
                 # Word to start at is words[trim_word_count]
                 # We re-find its position in the string (less error-prone than length math)
                 next_word = words[trim_word_count]
                 try:
                    # Find the first occurrence of the next word to be spoken
                    # The index returned will be in the *clean* text.
                    trim_index = clean_text.find(next_word)
                    if trim_index != -1:
                         # Now add the prefix length back to get the index in formatted_text
                         return trim_index + (len(text) - len(clean_text.lstrip()))
                 except:
                     pass # Fall through to length logic if word find fails

            # Fallback: just use character length up to the end of the last spoken word
            trim_index = len(" ".join(words[:trim_word_count]))
            
            # Find the actual character index in the formatted_text
            # Use `formatted_text.find(words[0])` to find the start of the clean text.
            start_of_clean_text = len(text) - len(clean_text.lstrip())
            
            # If the next word exists, find the next space after the trimmed part to ensure clean word break
            if trim_word_count < len(words):
                trimmed_text_part = clean_text[:trim_char_index_in_clean_text]
                # Now, find the index in the original string that corresponds to the end of the trimmed part
                # This is equivalent to: start_of_clean_text + len(trimmed_text_part)
                # We need to find the index of the first character of the word after the cut to be safe.
                
                # Best way: split by space, join, and measure the length.
                words_before_cut = " ".join(words[:trim_word_count])
                if words_before_cut:
                    # Find the index of the end of the last word spoken + the space after it
                    trim_idx = text.find(words_before_cut) + len(words_before_cut)
                    # Advance past any subsequent whitespace
                    while trim_idx < len(text) and text[trim_idx].isspace():
                        trim_idx += 1
                    return trim_idx
                else:
                    return start_of_clean_text # If 0 words trimmed, just return the start of the clean text
                
            else:
                 # If all words were trimmed, return the full length of the text.
                 return len(text)
                
            return 0 # Fallback
    
        
class BufferedPyAudioStreamer:
    """
    PyAudio-based streamer with 'Step-Down' buffering.
    - Starts with a large buffer (e.g. 10s) for safety.
    - Switches to a smaller buffer (e.g. 5s) to prevent silence on fast hardware
      while preventing micro-stutter on slow hardware.
    """
    def __init__(self, sample_rate=24000, buffer_duration=10.0, restart_at_garbage=False):
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_samples = int(sample_rate * buffer_duration)
        self.restart_at_garbage = restart_at_garbage
        
        # --- NEW SETTING: Secondary Buffer ---
        # Fast users won't notice it (latency is hidden by the queue).
        # Slow users will get 1s clean chunks instead of robotic stuttering.
        self.stream_chunk_duration = buffer_duration
        self.stream_threshold_samples = int(sample_rate * self.stream_chunk_duration)

        self.audio_queue = Queue()
        self.finished_flags = [False] 
        self.playing = False
        self.audio_thread = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.full_audio = np.array([], dtype=np.float32)
        
        # Flag to track if the massive initial buffer has been sent
        self.playback_started = False 

    def put(self, audio_chunk, indices):
        """Add audio chunk to the buffer and full audio collection"""
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.cpu().float().numpy()
        
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
            
        self.full_audio = np.concatenate([self.full_audio, audio_chunk]) if self.full_audio.size else audio_chunk
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk]) if self.audio_buffer.size else audio_chunk
                    
        # --- DYNAMIC THRESHOLD LOGIC ---
        if not self.playback_started:
            # Phase 1: Waiting for the Big Initial Buffer (User Setting, e.g. 10s)
            current_threshold = self.buffer_samples
        else:
            # Phase 2: Streaming Mode (Lower Threshold, e.g. 1s)
            current_threshold = self.stream_threshold_samples
                    
        if len(self.audio_buffer) >= current_threshold:
            play_chunk = self.audio_buffer
            self.audio_buffer = np.array([], dtype=np.float32)
            self.audio_queue.put(play_chunk)
            
            # Once we push the first chunk, we switch to streaming mode
            self.playback_started = True

    def end(self, indices=None):
        """Called by model.generate() after each chunk."""
        self.finished_flags[0] = True 
        
        # Force flush whatever is left, BUT ONLY IF NO GARBAGE WAS DETECTED
        if self.audio_buffer.size > 0:
            play_chunk = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
            self.audio_queue.put(play_chunk)
            self.playback_started = True

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
            logger.error(f"Warning: freeing memory: {e}. Maybe exllama is not loaded yet.")
    
    def split_by_sentence_enhanced(self, text):
        # Split by newlines OR by sentence terminators followed by whitespace
        sentences = re.split(r'\n|(?<=[.!?])\s+', text.strip())
        # Remove empty strings and strip whitespace
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
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
                from processor.vibevoice_processor import VibeVoiceProcessor
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
                             max_new_tokens = 32637, 
                             negative_llm_steps_to_skip = 0.0, 
                             semantic_steps_to_skip = 0.0, 
                             increase_cfg=False,
                             solver_order = 2,
                             audio_streamer: BufferedPyAudioStreamer = None, 
                             wav2lip_streamer: Wav2LipStreamer = None,
                             restart_at_garbage: bool = False,
                             use_compile: bool = False) -> dict:
        """Generate audio using VibeVoice model with optional streaming and returning full audio"""
        
        try:
            if self.model is None or self.processor is None:
                raise Exception("Model or processor not loaded")
                
            # Dynamically update scheduler order if requested
            if solver_order is not None:
                current_order = getattr(self.model.noise_scheduler.config, "solver_order", None)                
                if current_order != solver_order:
                    print(f"Switching scheduler order from {current_order} to {solver_order}")                    
                    # Re-instantiate the scheduler with the new order
                    new_config = dict(self.model.noise_scheduler.config)
                    new_config['solver_order'] = solver_order                    
                    self.model.noise_scheduler = DPMSolverMultistepScheduler.from_config(new_config)
                
            if use_compile:
                self.model.semantic_tokenizer = torch.compile(self.model.semantic_tokenizer, mode="reduce-overhead")            
                self.model.acoustic_tokenizer = torch.compile(self.model.acoustic_tokenizer, mode="reduce-overhead")            
                self.model.prediction_head = torch.compile(self.model.prediction_head, mode="reduce-overhead")                     
            
            # 1. Setup Streamer
            local_streamer = audio_streamer
            is_streaming_active = (local_streamer is not None) or streaming
            if local_streamer is None and streaming:
                local_streamer = BufferedPyAudioStreamer(sample_rate=24000, buffer_duration=buffer_duration)
                local_streamer.start_playback()   

            # 2. Initialize Retry Loop Variables
            current_text_to_speak = formatted_text
            current_seed = seed
            accumulated_waveform = None
            max_retries = 10
            retry_count = 0
            
            # --- START GENERATION LOOP ---
            while True:                
                # A. Prepare Inputs (Re-run every loop as text might change)
                torch.manual_seed(current_seed)
                np.random.seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
                
                self.model.negative_llm_steps_to_skip = negative_llm_steps_to_skip
                self.model.semantic_steps_to_skip = semantic_steps_to_skip
                self.model.increase_cfg = increase_cfg
                self.model.set_ddpm_inference_steps(diffusion_steps)

                inputs = self.processor([current_text_to_speak], voice_samples=[voice_samples], return_tensors="pt", return_attention_mask=True)
                device = next(self.model.parameters()).device
                inputs = {
                    k: v.to(device) if hasattr(v, 'to') else v 
                    for k, v in inputs.items()
                }

                # B. Setup Stop/Interrupt check
                stop_check_fn = None
                if INTERRUPTION_SUPPORT:
                    def final_stop_check():
                        import comfy.model_management as mm
                        mm.throw_exception_if_processing_interrupted()
                    stop_check_fn = final_stop_check

                # C. Run Inference
                with torch.no_grad():
                    gen_kwargs = {
                        "tokenizer": self.processor.tokenizer,
                        "cfg_scale": cfg_scale,
                        "max_new_tokens": max_new_tokens,
                        "audio_streamer": local_streamer,
                        "wav2lip_streamer": wav2lip_streamer,
                        "return_speech": True, # Always return speech to measure duration for trimming
                        "stop_check_fn": stop_check_fn,
                        "restart_at_garbage": restart_at_garbage,
                    }
                    if use_sampling:
                        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
                    else:
                        gen_kwargs["do_sample"] = False
                    
                    # CALL GENERATE
                    # Note: generate() must be updated to return partial audio and 'restart_needed' flag
                    output = self.model.generate(**inputs, **gen_kwargs)
                
                # D. Process Output of this attempt
                chunk_waveform = None
                speech_tensors = getattr(output, 'speech_outputs', None)
                
                if speech_tensors:
                    raw_waveform = speech_tensors[0] if isinstance(speech_tensors, list) else speech_tensors
                    if raw_waveform is not None:
                        if raw_waveform.dtype == torch.bfloat16:
                            raw_waveform = raw_waveform.to(torch.float32)
                        chunk_waveform = raw_waveform.cpu()
                        
                        # Accumulate valid audio into the master buffer
                        if accumulated_waveform is None:
                            accumulated_waveform = chunk_waveform
                        else:
                            # Concatenate along time dimension (dim -1)
                            accumulated_waveform = torch.cat([accumulated_waveform, chunk_waveform], dim=-1)

                # E. Check for Restart Signal
                if hasattr(output, 'restart_needed') and output.restart_needed:
                    if retry_count >= max_retries:
                        print(f"Max retries ({max_retries}) reached. Stopping generation.")
                        break
                    
                    print(f"Garbage audio detected in attempt {retry_count+1}. Preparing restart...")
                    
                    # dont use estimaation based on audio length, use estimaation based on avg ratio steps / words == 4.0
                    # so we need to get (last step number - noise steps) from generate() method and use it to estimate ok words
                    
                    # 1. Get the number of valid steps from the output
                    valid_steps = getattr(output, 'valid_steps_count', 0)
                    print(f"Detected {valid_steps} valid steps before garbage.")

                    # 2. Trim the text based on valid steps using the new metric (steps/word == 4.0)
                    trim_idx = estimate_word_trim_position(current_text_to_speak, valid_steps, steps_per_word=6.0)
                    
                    # 3. Update text for next run
                    # Assuming current_text_to_speak has the 'Speaker 0: ' prefix if it's not the first run
                    trimmed_part = current_text_to_speak[:trim_idx]
                    current_text_to_speak = "Speaker 0: " + current_text_to_speak[trim_idx:].strip()
                    print ("added Speaker 0: to text, todo: add check for single speaker.")
                    
                    if not current_text_to_speak.strip(): # Check if only 'Speaker 0: ' is left
                        print("Text fully spoken (or fully trimmed). Stopping.")
                        break
                        
                    print(f"Trimmed {len(trimmed_part)} chars ({trim_idx} index). Restarting with text: '{current_text_to_speak[:30]}...'")
                    
                    # 4. Update State for next loop
                    current_seed += 1 # CRITICAL: Change seed to avoid regenerating the exact same noise
                    retry_count += 1
                    
                    # Loop back to 'A' to start fresh inference with new text and seed
                    continue 
                
                # If we get here, generation finished successfully without noise.
                break 

            # --- END GENERATION LOOP ---

            # F. Final Cleanup and Return
            if is_streaming_active and local_streamer:
                if audio_streamer is None:
                    # If we created the streamer locally, we must close it and return the full buffer
                    local_streamer.end()
                    while local_streamer.playing:
                        time.sleep(0.1)
                    full_audio_np = local_streamer.get_full_audio()
                    waveform = torch.from_numpy(full_audio_np).unsqueeze(0).unsqueeze(0) if full_audio_np.size > 0 else None
                else:
                    # If streamer was passed in externally, just signal we are done with this segment
                    # The caller (single_speaker_node) handles closing.
                    return {"waveform": None, "sample_rate": 24000, "streamed": True}

                return {"waveform": waveform, "sample_rate": 24000, "streamed": True}
            else:
                # Non-streaming return: return the accumulated waveform from all attempts
                if accumulated_waveform is None:
                     # Fallback if something went wrong and nothing was generated
                     accumulated_waveform = torch.zeros(1, 1, int(0.5 * 24000))
                
                return {"waveform": accumulated_waveform.unsqueeze(0), "sample_rate": 24000, "streamed": False}
                    
        except Exception as e:
            # Handle interruptions or real errors
            logger.error(f"VibeVoice generation failed: {e}")
            if 'local_streamer' in locals() and local_streamer and audio_streamer is None:
                local_streamer.end()
            raise      
