import re
import json

class ReplaceStringMultipleNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "replace_json": ("STRING", {"default": '''{
"...": ", ",
"…": ", ",
"..": "." ,
"(": "" ,
")": "." ,
"  ": " ",
" ,": ",",
"*": "",
"—": "",
":\\n": ".\\n",
"\"": ""
}''', "multiline": True}),
                "output_format": (["single_speaker", "multiple_speakers"], {
                    "default": "single_speaker", 
                    "tooltip": "single_speaker - plain text. multiple_speakers - [1]: text"
                }),             
            },
            "optional": {
                "name_1": ("STRING", {"default": "author", "multiline": False}),
                "name_2": ("STRING", {"default": "", "multiline": False}),
                "name_3": ("STRING", {"default": "", "multiline": False}),
                "name_4": ("STRING", {"default": "", "multiline": False}),
                "name_5": ("STRING", {"default": "", "multiline": False}),
                "name_6": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    CATEGORY = "VibeVoiceWrapper"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    
    def execute(self, text, replace_json, output_format="single_speaker", name_1="author", name_2="", name_3="", name_4="", name_5="", name_6=""):
        try:
            # Build names list from input parameters
            input_names_list = [name_1]  # name_1 is always required (author)
            
            # Add optional names if they are not empty
            if name_2 and name_2.strip():
                input_names_list.append(name_2.strip())
            if name_3 and name_3.strip():
                input_names_list.append(name_3.strip())
            if name_4 and name_4.strip():
                input_names_list.append(name_4.strip())
            if name_5 and name_5.strip():
                input_names_list.append(name_5.strip())
            if name_6 and name_6.strip():
                input_names_list.append(name_6.strip())    
            
            # Convert list to comma-separated string for parse_names_to_ints
            input_names_str = ",".join(input_names_list)
            
            parsed_text = text.replace("**", "")
            
            if (output_format == "multiple_speakers"):
                # parse names to numbers
                parsed_text = self.parse_names_to_ints(input_names_str, text)
            
            # Second: apply JSON replacements
            replacements_dict = json.loads(replace_json)
            result = parsed_text
            for needle, replacement in replacements_dict.items():
                result = result.replace(needle, replacement)
            
            # Clean up the result: trim and replace trailing comma with dot
            result = result.strip()           
            
            return (result,)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return (text,)
    
    def parse_names_to_ints(self, input_names, text):
        """
        Parse names in dialogue to numbers [1]-[6]
        
        Args:
            input_names: comma-separated names (author is always first/1)
            text: input text with dialogue
            
        Returns:
            modified text with names replaced by numbers
        """
        # Parse input names
        names_list = []
        if input_names and input_names.strip():
            names_list = [name.strip() for name in input_names.split(',')]
        
        # Ensure we have at least author (index 1)
        if not names_list:
            names_list = ["author"]
        
        # Create name to number mapping (max 6 voices: 1-6)
        name_to_num = {}
        for i, name in enumerate(names_list):
            if i < 6:  # Only support up to 6 voices (1-6)
                name_to_num[name] = i + 1  # Shift indices by 1        
        
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            original_line = line.strip(" ,—-")
            if not line:
                # Skip empty lines or keep them as empty
                result_lines.append("")
                continue
            
            # Check if line starts with any known name
            found_speaker = False
            for name, num in name_to_num.items():
                # Pattern to match:
                # ^(name)
                # (?: \s* (\([^)]*\)) )?  <- Optional action in parentheses
                # (?: \s* : )?           <- Optional colon (non-capturing, but we check for its presence)
                # (?: \s* (\([^)]*\)) )?  <- Optional action after colon

                # New pattern:
                # Group 1: The name itself
                # Group 2: Optional action before a potential colon
                # Group 3: The colon itself (if present)
                # Group 4: Optional action after a potential colon
                pattern = rf'^{re.escape(name)}(?:\s*(\([^)]*\)))?(\s*:)?(?:\s*(\([^)]*\)))?'
                match = re.match(pattern, original_line)

                if match:
                    colon_present = bool(match.group(2)) # True if match.group(2) is not None/empty

                    action_text_g1 = match.group(1) # Action before colon
                    action_text_g3 = match.group(3) # Action after colon

                    action_text = ""
                    if action_text_g1:
                        action_text = action_text_g1
                    elif action_text_g3: # Prioritize action after colon if both present, or if only after
                        action_text = action_text_g3

                    # Clean and capitalize action text
                    if action_text:
                        action_text = action_text.strip(" ,—-\n")
                        if action_text.startswith("("): # If it's an action in parentheses, capitalize the first letter inside
                            if len(action_text) > 1: # Ensure there's content to capitalize
                                action_text = "(" + action_text[1].upper() + action_text[2:]
                        else: # If it's just text, capitalize the first letter
                            action_text = action_text[0].upper() + action_text[1:]

                    # Determine the remaining text and how to construct the output
                    if colon_present:
                        # If colon is present, strip the name and its associated action/colon part
                        remaining_text = original_line[match.end():].strip(" ,—-\n")

                        # Capitalize the first letter of the remaining text if it's not empty
                        if remaining_text:
                            remaining_text = remaining_text[0].upper() + remaining_text[1:]

                        # Build the new line: [num]: (Action) Remaining Text
                        if action_text:
                            new_line = f"[{num}]: {action_text} {remaining_text}" # Keep colon after action if it was originally there
                        else:
                            new_line = f"[{num}]: {remaining_text}" # No action, just remaining text after colon
                    else:
                        # No colon, keep the name and its associated action in the text
                        # The line should be exactly as it was, but with [num]: prepended
                        new_line = f"[{num}]: {original_line}"
                        # However, if there was an action *before* the name (e.g., Лида (подмигивает)),
                        # we need to ensure the action is part of the 'original_line' that we are keeping.
                        # The original_line already includes it, so this is fine.
                        # The only special case is the example: Лида (подмигивает) подходит к остальным.
                        # The output for this is: [1]: Лида (подмигивает) подходит к остальным.
                        # So we just prepend the [num]: to the original line.

                    result_lines.append(new_line)
                    found_speaker = True
                    break
            
            if not found_speaker:
                # Line doesn't start with a known name - treat as author's note
                result_lines.append(f"[1]: {line}")
        
        return '\n'.join(result_lines)

# For backward compatibility with your existing workflow
NODE_CLASS_MAPPINGS = {
    'Replace String Multiple with json': ReplaceStringMultipleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Replace String Multiple with json': 'Replace String Multiple using JSON'
}