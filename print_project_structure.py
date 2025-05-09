import os

def print_tree(startpath, prefix=""):
    items = os.listdir(startpath)
    items.sort()
    for index, item in enumerate(items):
        path = os.path.join(startpath, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if index == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

# Set your base directory here
base_directory = os.getcwd()

print(f"📂 Project Structure: {os.path.basename(base_directory)}\n")
print_tree(base_directory)
