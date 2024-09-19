"""
Script to check model registry entries for regEx in EOS culling strings.
Run this script after adding new model entries for the HuggingFace, llama-cpp and vLLM backends. These backends use
re.sub to cull EOS and other sequences at the end of model outputs flexibly - but this means that characters/substrings
that can be parsed as special python regEx special characters/sequences need to be properly escaped if they are part of
the model's special tokens and/or chat template.
"""

import json
import os

# all python regEx characters/sequences:
regex_specials = [".", "^", "$", "*", "+", "?", "{", "}", ",", "[", "]", "|", "(", ")"]
regex_escape = "\\"
regex_sequences = [r"\A", r"\b", r"\B", r"\d", r"\D", r"\s", r"\S", r"\w", r"\W", r"\Z"]


def check_model_registry_eos(model_registry_path):
    # load model registry:
    with open(model_registry_path, 'r', encoding='utf-8') as registry_file:
        model_registry = json.load(registry_file)

    # check all entries for regEx in EOS:
    for model_entry in model_registry:
        regex_found = False
        unescaped_regex = False
        unescaped_sequences = list()
        if "eos_to_cull" in model_entry:
            model_name = model_entry["model_name"]
            eos = model_entry["eos_to_cull"]

            if regex_escape in eos:
                regex_found = True

            for idx, char in enumerate(eos):
                if char in regex_specials:
                    regex_found = True
                    # check if special character is escaped:
                    if not eos[idx - 1] == regex_escape:
                        unescaped_regex = True
                        unescaped_sequences.append((char, idx, repr(eos[idx - 1:idx + 1])))

            for regex_sequence in regex_sequences:
                if regex_sequence in eos:
                    # print(f"Special regEx sequence '{regex_sequence}' in EOS!")
                    regex_found = True

            if regex_found:
                print(f"{model_name} for {model_entry['backend']} has regEx EOS: {repr(eos)}")
                if unescaped_regex:
                    print(f"There is unescaped regEx in this EOS, make sure this is intentional:")
                    for unescaped_sequence in unescaped_sequences:
                        print(f"Special character '{unescaped_sequence[0]}' at position {unescaped_sequence[1]}, "
                              f"pair {unescaped_sequence[2]}")
                print()


check_model_registry_eos("model_registry.json")

if "model_registry_custom.json" in os.listdir():
    print("Custom model registry found, checking for regEx EOS...")
    print()
    check_model_registry_eos("model_registry_custom.json")
