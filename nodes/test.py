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
                "replace_json": ("STRING", {"default": '{"old":"new"}', "multiline": True}),
            }
        }
    
    CATEGORY = "VibeVoiceWrapper"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    
    def execute(self, text, replace_json):
        try:
            # Parse the JSON string into a dictionary
            replacements_dict = json.loads(replace_json)
            
            # Apply replacements
            result = text
            for needle, replacement in replacements_dict.items():
                # Simple string replacement
                result = result.replace(needle, replacement)
            
            return (result,)
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return (text,)
    
    @classmethod
    def parse_names_to_ints(cls, input_names, text):
        """
        Parse names in dialogue to numbers [0]-[3]
        
        Args:
            input_names: comma-separated names (author is always first/0)
            text: input text with dialogue
            
        Returns:
            modified text with names replaced by numbers
        """
        # Parse input names
        names_list = []
        if input_names and input_names.strip():
            names_list = [name.strip() for name in input_names.split(',')]
        
        # Ensure we have at least author (index 0)
        if not names_list:
            names_list = ["author"]
        
        # Create name to number mapping (max 4 voices: 0-3)
        name_to_num = {}
        for i, name in enumerate(names_list):
            if i < 4:  # Only support up to 4 voices (0-3)
                name_to_num[name] = i
        
        lines = text.split('\n')
        result_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # Skip empty lines or keep them as empty
                result_lines.append("")
                continue
            
            # Check if line starts with any known name
            found_speaker = False
            for name, num in name_to_num.items():
                # Pattern to match name at start of line, possibly followed by space and (action)
                pattern = rf'^{re.escape(name)}\s*(\([^)]*\))?\s*:'
                match = re.match(pattern, line)
                if match:
                    # Extract the rest of the line after name and colon
                    remaining_text = line[match.end():].strip()
                    
                    # Build the replacement line
                    if match.group(1):  # If there's an action in parentheses
                        new_line = f"[{num}]: {match.group(1)}: {remaining_text}"
                    else:
                        new_line = f"[{num}]: {remaining_text}"
                    
                    result_lines.append(new_line)
                    found_speaker = True
                    break
            
            if not found_speaker:
                # Line doesn't start with a known name - treat as author's note
                result_lines.append(f"[0]: {line}")
        
        return '\n'.join(result_lines)

# For backward compatibility with your existing workflow
NODE_CLASS_MAPPINGS = {
    'Replace String Multiple with json': ReplaceStringMultipleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Replace String Multiple with json': 'Replace String Multiple using JSON'
}

# Example usage
if __name__ == "__main__":
    node = ReplaceStringMultipleNode()
    
    input_names = "author, Саша, Сергей, Анна"
    input_text = '''Саша: "Сергей, я думаю, что наказание должно быть понятным и эффективным. Это будет наглядным уроком для всех."

Сергей (с удивлением): "Один день? Ты уверена, Саша? Это же строгое наказание."

Саша (немного смущенно, но решительно): "Да, это строго, но это также будет наглядным примером для других. Они должны понять, что нарушение правил недопустимо.

Сергей: "Ты действительно умная, Саша. Ладно, давай так и сделаем"

Саша и Сергей идут к комнатам Анны и Алины. Они заходят в комнату Анны первой.

Анна (сидит на кровати, смущенная и испуганная): *"Сергей, Саша… я…"'''

    result = node.parse_names_to_ints(input_names, input_text)
    print("Input:")
    print(input_text)
    print("\nOutput:")
    print(result)