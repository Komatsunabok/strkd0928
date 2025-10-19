import os

# モデルフォルダの定義
base_folders = {
    "Students": "save/students/models",
    "Teachers": "save/teachers/models"
}

output_file = "model_list.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for role, folder in base_folders.items():
        f.write(f"=== {role} ===\n")
        if os.path.exists(folder):
            # 下層フォルダ（モデル名）を取得
            for name in os.listdir(folder):
                path = os.path.join(folder, name)
                if os.path.isdir(path):  # フォルダのみ対象
                    f.write(name + "\n")
        else:
            f.write(f"{folder} not found.\n")
        f.write("\n")

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
