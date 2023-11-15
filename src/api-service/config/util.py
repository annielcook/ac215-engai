import json

def convert_json_config() -> None:
    """
    Helper function to convert breed-to-index
    mapping --> index-to-breed mapping."""
    with open('breed-to-index.json', 'r') as f:
        in_dict = json.load(f)
    out_dict = {v: k for k, v in in_dict.items()}
    
    with open('index-to-breed.json', 'w') as f:
        json.dump(out_dict, f)


if __name__ == '__main__':
    convert_json_config()