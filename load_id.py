import os
def load_indices(filename):
    indices = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
            # Replace common separators with comma
            content = content.replace('\n', ',').replace('.', ',')
            parts = content.split(',')
            for p in parts:
                p = p.strip()
                if p.isdigit():
                    indices.append(int(p))
    else:
        print(f"Warning: File '{filename}' not found.")
    return indices