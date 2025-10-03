import os

# チェックしたいフォルダを指定
folder_path = r"C:\Users\quter\OneDrive - 筑波大学\Adapt\研究\program\KD\strkd0928"

# 100MB をバイトに変換
threshold = 100 * 1024 * 1024  

print(f"Checking files larger than 100MB in: {folder_path}\n")
print("===!!!WARNING!!!===")
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            size = os.path.getsize(file_path)
            if size > threshold:
                print(f"{file_path} : {size / (1024*1024):.2f} MB")
        except OSError as e:
            print(f"Error accessing {file_path}: {e}")
