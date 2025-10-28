import os

# 探索対象のルートディレクトリ（例: プロジェクトのルート）
root_dir = "."
gitignore_path = os.path.join(root_dir, ".gitignore")
size_threshold = 100 * 1024 * 1024  # 100MB

whitelist = [
    ".git/objects/pack/pack-71940e191c0fdf61ba8814ce1e06abbef62770b6.pack",
    ".git/objects/pack/",
]

# 既存の.gitignoreを読み込む
if os.path.exists(gitignore_path):
    with open(gitignore_path, "r", encoding="utf-8") as f:
        existing_ignores = {line.strip() for line in f if line.strip() and not line.startswith("#")}
else:
    existing_ignores = set()

files_to_ignore = []

# ファイルを探索
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        try:
            size = os.path.getsize(file_path)
        except OSError:
            continue

        if size >= size_threshold:
            rel_path = os.path.relpath(file_path, root_dir).replace("\\", "/")

            if any(rel_path.startswith(w) for w in whitelist):
                print(f"[-] Skipped whitelisted file: {rel_path}")
                continue

            if rel_path not in existing_ignores:
                files_to_ignore.append((rel_path, size))
                print(f"[+] {rel_path} ({size / (1024 * 1024):.2f} MB)")

        # if size >= size_threshold:
        #     rel_path = os.path.relpath(file_path, root_dir).replace("\\", "/")
        #     if rel_path not in existing_ignores:
        #         files_to_ignore.append((rel_path, size))
        #         print(f"[+] {rel_path} ({size / (1024 * 1024):.2f} MB)")

# 追加があれば.gitignoreを更新
if files_to_ignore:
    with open(gitignore_path, "a", encoding="utf-8") as f:
        f.write("\n\n# Automatically added large files\n")
        for rel_path, size in files_to_ignore:
            f.write(rel_path + "\n")
    print(f"\n✅ Added {len(files_to_ignore)} large files to .gitignore.")
else:
    print("\nNo new large files found or all already ignored.")
