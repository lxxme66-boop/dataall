import os

def rename_md_to_txt(folder_path):
    """
    递归将 folder_path 中所有 .md 文件改名为 .txt 文件（不保留原 .md 文件）
    """
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".md"):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, os.path.splitext(file)[0] + ".txt")
                os.rename(old_path, new_path)
                count += 1
                print(f"[改名] {old_path} -> {new_path}")
    print(f"\n✅ 共改名 {count} 个 .md 文件为 .txt 文件")

if __name__ == "__main__":
    folder = "origedata"  # 改成服务器上的文件夹路径
    rename_md_to_txt(folder)
