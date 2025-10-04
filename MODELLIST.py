import os

folders = ["save/download", "save/students", "save/teachers"]
output_file = "model_list.txt"

with open(output_file, "w") as f:
    for folder in folders:
        if os.path.exists(folder):
            f.write(f"=== {folder} ===\n")
            for name in os.listdir(folder):
                f.write(name + "\n")
            f.write("\n")
        else:
            f.write(f"{folder} not found.\n\n")

print(f"モデル一覧を {output_file} に保存しました！")

import os

def list_tree(root, prefix=""):
    entries = os.listdir(root)
    entries.sort()
    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        yield prefix + connector + entry
        if os.path.isdir(path):
            extension = "    " if i == len(entries) - 1 else "│   "
            yield from list_tree(path, prefix + extension)

output_file = "model_tree.txt"
root_dir = "save"

print(root_dir)
for line in list_tree(root_dir):
    print(line)
    

with open(output_file, "w", encoding="utf-8") as f:
    f.write(root_dir + "\n")
    for line in list_tree(root_dir):
        f.write(line + "\n")

print(f"ツリー構造の一覧を {output_file} に保存しました！")
