import os
from collections import defaultdict

# 你要统计的扩展名
exts = [".py", ".cu",  ".h", ".cpp", ".sh"]

# 初始化：每种扩展名的行数总计
line_counts = defaultdict(int)
file_counts = defaultdict(int)

for root, _, files in os.walk("."):
    for file in files:
        for ext in exts:
            if file.endswith(ext):
                try:
                    full_path = os.path.join(root, file)
                    # 如果路径中包含 third_party，则跳过该文件
                    if "third_party" in full_path:
                        continue
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        line_count = len(lines)
                        line_counts[ext] += line_count
                        file_counts[ext] += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")

# 输出结果
print("📊 Line Count by File Type:\n")
total = 0
for ext in exts:
    print(f"{ext:5} | {file_counts[ext]:4d} files | {line_counts[ext]:7d} lines")
    total += line_counts[ext]

print(f"\n🧮 Total lines across all: {total}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import os
from collections import defaultdict

def run_git_log(author):
    """运行 git log 命令并返回每次提交的 numstat 信息"""
    cmd = ["git", "log", f"--author={author}", "--pretty=tformat:", "--numstat", "--", "."]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print("Error running git log:", result.stderr)
        exit(1)
    return result.stdout.splitlines()

def parse_numstat(lines):
    """解析 numstat 格式的输出，返回按文件统计的增加/删除行数"""
    file_stats = defaultdict(lambda: {"added": 0, "deleted": 0})
    total_added = 0
    total_deleted = 0

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue  # 非 numstat 行
        added, deleted, filename = parts
        if added == '-' or deleted == '-':
            continue  # 二进制文件等无法统计
        added = int(added)
        deleted = int(deleted)
        file_stats[filename]["added"] += added
        file_stats[filename]["deleted"] += deleted
        total_added += added
        total_deleted += deleted

    return file_stats, total_added, total_deleted

def count_commits(author):
    """统计提交次数"""
    cmd = ["git", "log", f"--author={author}", "--pretty=oneline", "--", "."]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return len(result.stdout.strip().splitlines())

def main():
    author = "Ooosn"  # 修改为你要统计的作者名

    print(f"统计 Git 作者 '{author}' 在当前目录的贡献：\n")

    lines = run_git_log(author)
    file_stats, total_added, total_deleted = parse_numstat(lines)
    try:
        commit_count = count_commits(author)
    except:
        commit_count = 0
    print(f"总提交次数：{commit_count}")
    print(f"总增加行数：{total_added}")
    print(f"总删除行数：{total_deleted}\n")



if __name__ == "__main__":
    main()
