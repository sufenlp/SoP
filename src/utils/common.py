import ast
import logging
from fastchat.model import get_conversation_template


def extract_json(s):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `adversarial prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    # Extract the string that looks like a JSON
    start_pos = s.find("{")
    end_pos = s.find("}") + 1  # +1 to include the closing brace
    if end_pos == -1:
        logging.error("Error extracting potential JSON structure")
        logging.error(f"Input:\n {s}")
        return None, None

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n", "")  # Remove all line breaks

    try:
        parsed = ast.literal_eval(json_str)
        if not all(x in parsed for x in ["improvement", "prompt"]):
            logging.error("Error in extracted structure. Missing keys.")
            logging.error(f"Extracted:\n {json_str}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {json_str}")
        return None, None


def conv_template(template_name):
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template


if __name__ == '__main__':
    # conv = get_conversation_template('mistral')
    # conv.append_message(conv.roles[0], "hahaha")
    # conv.set_system_message("hahaha")
    # print(conv.get_prompt())

    print('-'*50)
    conv = get_conversation_template('Tulu 2')
    conv = get_conversation_template('tulu-2-dpo-70b')
    conv = get_conversation_template('tulu-2-dpo-7b')
    conv.append_message(conv.roles[0], "hahaha")
    conv.append_message(conv.roles[1], "hahaha")
    conv.set_system_message("hahaha")
    print(conv.get_prompt())

    print('-'*50)
    conv = get_conversation_template('Olmo')
    conv.append_message(conv.roles[0], "hahaha")
    conv.append_message(conv.roles[1], "hahaha")
    conv.set_system_message("hahaha")
    print(conv.get_prompt())

