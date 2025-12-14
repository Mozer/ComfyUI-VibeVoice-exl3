# Created by Fabio Sarracino
__version__ = "1.3.0"
__author__ = "Fabio Sarracino"
__title__ = "VibeVoice ComfyUI"

import logging
import os
import sys
import subprocess

# Setup logging
logger = logging.getLogger("VibeVoice")
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[VibeVoice] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def apply_timm_compatibility_patches():
    """Apply compatibility patches for timm package conflicts"""
    try:
        import timm.data
        
        # Patch missing functions that cause import errors
        patches = {
            'ImageNetInfo': lambda: type('ImageNetInfo', (), {'__init__': lambda self: None})(),
            'infer_imagenet_subset': lambda class_to_idx: 'imagenet',
            'get_imagenet_subset_labels': lambda *args, **kwargs: [],
            'get_imagenet_subset_info': lambda *args, **kwargs: {},
            'resolve_data_config': lambda *args, **kwargs: {}
        }
        
        for attr_name, patch_func in patches.items():
            if not hasattr(timm.data, attr_name):
                if attr_name == 'ImageNetInfo':
                    setattr(timm.data, attr_name, type('ImageNetInfo', (), {'__init__': lambda self: None}))
                else:
                    setattr(timm.data, attr_name, patch_func)
        
        return True
    except Exception as e:
        return False

def check_embedded_vibevoice():
    """Check if embedded VibeVoice is available"""
    vvembed_path = os.path.join(os.path.dirname(__file__), 'vvembed')
    if not os.path.exists(vvembed_path):
        logger.error(f"Embedded VibeVoice not found at {vvembed_path}")
        return False
    
    # Add vvembed to path if not already there
    if vvembed_path not in sys.path:
        sys.path.insert(0, vvembed_path)
    
    logger.info("Using embedded VibeVoice (MIT licensed)")
    return True

def ensure_dependencies():
    """Ensure required dependencies are installed"""
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.44.0"):
            logger.warning("Transformers version < 4.44.0, some features may not work correctly")
    except ImportError:
        logger.warning("Transformers not installed. Please install: pip install transformers>=4.44.0")
        return False
    
    # Apply timm patches if needed
    apply_timm_compatibility_patches()
    
    return True

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


# Register VibeVoice nodes (using embedded VibeVoice)
if check_embedded_vibevoice() and ensure_dependencies():
    try:
        from .nodes.load_text_node import LoadTextFromFileNode
        from .nodes.replace_multiple_node import ReplaceStringMultipleNode
        from .nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
        from .nodes.multi_speaker_node import VibeVoiceMultipleSpeakersNode
        from .nodes.free_memory_node import VibeVoiceFreeMemoryNode   
        
        NODE_CLASS_MAPPINGS["LoadTextFromFileNode"] = LoadTextFromFileNode
        NODE_DISPLAY_NAME_MAPPINGS["LoadTextFromFileNode"] = "VibeVoice Load Text From File"
        
        NODE_CLASS_MAPPINGS["ReplaceStringMultipleNode"] = ReplaceStringMultipleNode
        NODE_DISPLAY_NAME_MAPPINGS["ReplaceStringMultipleNode"] = "Replace String Multiple using json"             
        
        # Single speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceSingleSpeakerNodeExl3"] = VibeVoiceSingleSpeakerNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceSingleSpeakerNodeExl3"] = "VibeVoice Single Speaker Exl3"
        
        # Multi speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceMultipleSpeakersNodeExl3"] = VibeVoiceMultipleSpeakersNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceMultipleSpeakersNodeExl3"] = "VibeVoice Multiple Speakers Exl3"
        
        # Free memory node
        NODE_CLASS_MAPPINGS["VibeVoiceFreeMemoryNodeExl3"] = VibeVoiceFreeMemoryNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceFreeMemoryNodeExl3"] = "VibeVoice Free Memory Exl3"
        
        logger.info("VibeVoice nodes registered successfully")
        
    except Exception as e:
        logger.error(f"Failed to register VibeVoice nodes: {e}")
        logger.info("Please ensure transformers>=4.44.0 is installed")
else:
    logger.warning("VibeVoice nodes unavailable - check embedded module and dependencies")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']