import torch

class OscillationDetector:
    def __init__(self, window_size=25, stability_threshold=0.005):
        """
        Args:
            window_size (int): Number of chunks to keep in history. 
                               25 chunks * 0.11s ~= 2.75 seconds.
            stability_threshold (float): If the standard deviation of the HISTORY 
                                         is below this, the signal is too stable (beep).
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.history = []

    def is_noise(self, audio_chunk):
        """
        Checks if the current audio chunk is part of a noise/beep pattern.
        """
        # 1. Calculate the 'loudness' (Standard Deviation) of the current chunk
        # .std() returns a 0-d tensor. We keep it as a tensor to stay on GPU.
        chunk_std = audio_chunk.std()

        # 2. Append to history list
        self.history.append(chunk_std)

        # 3. Maintain window size
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 4. Check for pattern (only if we have enough history)
        if len(self.history) == self.window_size:
            # Stack the list of scalar tensors into a 1D tensor [window_size]
            # This operation is very fast on GPU
            history_tensor = torch.stack(self.history)
            
            # Calculate the deviation of the history itself
            # High value = Speech (loudness jumps up and down)
            # Low value = Beep/Drone (loudness is constant or changes very slowly)
            history_fluctuation = history_tensor.std()

            if history_fluctuation < self.stability_threshold:
                return True

        return False

    def reset(self):
        """Clear history between inference calls"""
        self.history = []

class GarbageAudioException(Exception):
    """Custom exception to stop generation cleanly"""
    pass