import sys
import yaml

def merge_dicts(source, destination):
    """
    Recursively merges two dictionaries.
    Attributes from source are added to destination only if they are not already present.
    If an attribute exists in both and both are dictionaries, a recursive merge is performed.
    """
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
        elif isinstance(value, dict) and isinstance(destination[key], dict):
            merge_dicts(value, destination[key])
    return destination

def main():
    if len(sys.argv) != 3:
        print("Usage: merge_yaml.py <example_file> <dest_file>")
        sys.exit(1)

    example_path = sys.argv[1]
    dest_path = sys.argv[2]

    try:
        with open(example_path, 'r') as f:
            example_data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Error reading example file: {e}")
        sys.exit(1)

    try:
        with open(dest_path, 'r') as f:
            dest_data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # If the destination file does not exist, just use the data from the example
        dest_data = example_data
    except Exception as e:
        print(f"Error reading destination file: {e}")
        sys.exit(1)
    else:
        # If the file exists, perform a merge
        dest_data = merge_dicts(example_data, dest_data)

        # Remove tracer and http-client components if present
        components = dest_data.get('components_manager', {}).get('components')
        if isinstance(components, dict):
            components.pop('tracer', None)
            components.pop('http-client', None)

    try:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, 'w') as f:
            yaml.dump(dest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"Error writing to destination file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
