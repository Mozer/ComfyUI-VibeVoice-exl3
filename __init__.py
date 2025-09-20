# Created by Fabio Sarracino
__version__ = "1.0.3"
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

def check_vibevoice_available():
    """Check if VibeVoice is available for import"""
    try:
        # Apply timm patches first
        apply_timm_compatibility_patches()
        
        import vibevoice
        return True
    except ImportError:
        return False

def install_vibevoice():
    """Install VibeVoice if not already installed"""
    if check_vibevoice_available():
        return True
        
    try:
        # Install VibeVoice with specific transformers version to avoid LossKwargs issue
        logger.info("Installing VibeVoice with compatible dependencies...")
        
        # First install compatible transformers version
        transformers_cmd = [sys.executable, "-m", "pip", "install", "transformers>=4.44.0"]
        result = subprocess.run(transformers_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.warning(f"Transformers install warning: {result.stderr}")
        
        # Then install VibeVoice
        cmd = [sys.executable, "-m", "pip", "install", "git+https://github.com/microsoft/VibeVoice.git"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("VibeVoice installation completed")
            return check_vibevoice_available()  # Verify installation
        else:
            logger.error(f"Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Installation error: {e}")
        return False

# Initialize node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register text loading node (always available)
try:
    from .nodes.load_text_node import LoadTextFromFileNode
    NODE_CLASS_MAPPINGS["LoadTextFromFileNode"] = LoadTextFromFileNode
    NODE_DISPLAY_NAME_MAPPINGS["LoadTextFromFileNode"] = "VibeVoice Load Text From File"
except Exception as e:
    logger.error(f"Failed to register LoadTextFromFile node: {e}")

# Register VibeVoice nodes (requires VibeVoice installation)
if install_vibevoice():
    try:
        from .nodes.single_speaker_node import VibeVoiceSingleSpeakerNode
        from .nodes.multi_speaker_node import VibeVoiceMultipleSpeakersNode
        from .nodes.free_memory_node import VibeVoiceFreeMemoryNode
        
        # Single speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceSingleSpeakerNodeExl3"] = VibeVoiceSingleSpeakerNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceSingleSpeakerNodeExl3"] = "VibeVoice Single Speaker Exl3"
        
        # Multi speaker node
        NODE_CLASS_MAPPINGS["VibeVoiceMultipleSpeakersNodeExl3"] = VibeVoiceMultipleSpeakersNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceMultipleSpeakersNodeExl3"] = "VibeVoice Multiple Speakers Exl3"
        
        # Free memory node
        NODE_CLASS_MAPPINGS["VibeVoiceFreeMemoryNodeExl3"] = VibeVoiceFreeMemoryNode
        NODE_DISPLAY_NAME_MAPPINGS["VibeVoiceFreeMemoryNodeExl3"] = "VibeVoice Free Memory Exl3"
        
    except Exception as e:
        logger.error(f"Failed to register VibeVoice nodes: {e}")
        logger.info("VibeVoice may need a ComfyUI restart to complete installation")
else:
    logger.warning("VibeVoice nodes unavailable - installation failed")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', '__version__']