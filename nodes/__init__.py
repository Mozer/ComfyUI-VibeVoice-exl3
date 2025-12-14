# Created by Fabio Sarracino
# Nodes module for VibeVoiceWrapper
"""
This module contains all the ComfyUI nodes for VibeVoice integration.
"""

from .load_text_node import LoadTextFromFileNode
from .replace_multiple_node import ReplaceStringMultipleNode
from .single_speaker_node import VibeVoiceSingleSpeakerNode
from .multi_speaker_node import VibeVoiceMultipleSpeakersNode
from .free_memory_node import VibeVoiceFreeMemoryNode

__all__ = [
    'LoadTextFromFileNode', 
    'VibeVoiceSingleSpeakerNode', 
    'VibeVoiceMultipleSpeakersNode',
    'VibeVoiceFreeMemoryNode',
    'ReplaceStringMultipleNode',
]