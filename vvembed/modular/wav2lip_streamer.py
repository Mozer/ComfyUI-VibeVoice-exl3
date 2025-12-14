import os
import torch
import numpy as np
import threading
import requests
from scipy.io import wavfile

class Wav2LipStreamer:

    # Class variable shared across all instances
    audio_chunk_counter = 1
    reply_part_counter = 0
    class_lock = threading.Lock()
    
    """Streamer that saves audio chunks to disk for wav2lip processing"""
    def __init__(self, save_path="c:/DATA/LLM/SillyTavern-extras/tts_out/", char_name="default", sample_rate=24000):
        self.save_path = save_path
        self.char_name = char_name
        self.sample_rate = sample_rate
        # Get and increment the class counter
        with Wav2LipStreamer.class_lock:
            self.audio_chunk_i = Wav2LipStreamer.audio_chunk_counter
            self.reply_part_i = Wav2LipStreamer.reply_part_counter
            Wav2LipStreamer.audio_chunk_counter += 1
            Wav2LipStreamer.reply_part_counter += 1
        self.buffer = np.array([], dtype=np.float32)
        self.samples_per_chunk = 3*sample_rate  # 3 seconds at 24kHz
        self.api_called = False
        self.lock = threading.Lock()
        
        # Ensure save directory exists
        os.makedirs(save_path, exist_ok=True)

    def put(self, audio_chunk):
        """Add audio chunk to buffer and save to disk when we have 1 second of audio"""
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.cpu().float().numpy()
        
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze()
            
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, audio_chunk]) if self.buffer.size else audio_chunk        
                
        # Process complete 1-second chunks
        while len(self.buffer) >= self.samples_per_chunk:
            self._save_chunk()
    
    def _save_chunk(self):
        """Save 1-second chunk to disk and call API"""
        with self.lock:
            # Extract 1-second chunk
            chunk_to_save = self.buffer[:self.samples_per_chunk]
            self.buffer = self.buffer[self.samples_per_chunk:]
            
            # Generate filename
            current_wav = f"out_{self.audio_chunk_i}"
            filename = os.path.join(self.save_path, current_wav + ".wav")
            
            # Save as 16-bit PCM WAV file
            int_audio = np.int16(chunk_to_save * 32767)
            wavfile.write(filename, self.sample_rate, int_audio)
            
            # Call wav2lip API
            self._call_wav2lip_api(self.audio_chunk_i, self.reply_part_i)
            
            # Increment for next chunk
            with Wav2LipStreamer.class_lock:
                self.audio_chunk_i = Wav2LipStreamer.audio_chunk_counter
                Wav2LipStreamer.audio_chunk_counter += 1
            
            # Increment reply part
            with Wav2LipStreamer.class_lock:
                self.reply_part_i = Wav2LipStreamer.reply_part_counter
                Wav2LipStreamer.reply_part_counter += 1            
    
    def _call_wav2lip_api(self, current_audio_chunk_i, reply_part=0):
        """Asynchronously call wav2lip API"""
        def api_call():
            try:
                print("wav2lip API called")
                current_wav = f"out_{current_audio_chunk_i}"
                # current_audio_chunk_i (1-9999): start from 1 and increment globally
                # reply_part (0-9999): start from 0 and increment at each sentence. 0 - new char reply came (first sentence), drop playing current reply if any
                url = f"http://127.0.0.1:5100/api/wav2lip/generate/{self.char_name}/cuda/{current_wav}/latest/{current_audio_chunk_i}/{reply_part}/None"
                requests.get(url, timeout=1)  # Short timeout since we don't need response
            except requests.exceptions.RequestException:
                # Silently fail - wav2lip might not be ready yet
                pass
        
        # Start API call in background thread
        thread = threading.Thread(target=api_call)
        thread.daemon = True
        thread.start()
    
    def flush(self):
        """Save any remaining audio in buffer"""
        print("wav2lip flush")
        if len(self.buffer) > 0:
            self._save_chunk()
            
    def reset_reply_part(self):
        # reset reply_part to 0
        print("reset reply part to 0")
        with Wav2LipStreamer.class_lock:
            self.reply_part_i = 0
            Wav2LipStreamer.reply_part_counter = 0  