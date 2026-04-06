import time, threading, os, re

filename = r"C:\Users\WANG\Documents\xwechat_files\wxid_jchazb9ye38e11_1dfc\msg\file\2025-10\43244.txt"   # 替换成你的小说路径
encoding = "utf-8"
record_file = f"{filename}.record"
chapter_pattern = re.compile(
    r'^\s*(?:'
    r'第\s*[0-9０-９一二三四五六七八九十百千两〇零○]+\s*(?:章|节|回|卷)'
    r'|(?:Chapter|CHAPTER)\s*[0-9０-９]+'
    r')\s*[：:.\-\s]*', 
    re.IGNORECASE
)

# 加载文本
try:
    with open(filename, 'r', encoding=encoding) as f:
        lines = f.readlines()
except UnicodeDecodeError:
    with open(filename, 'r', encoding='gbk') as f:
        lines = f.readlines()

# 提取章节索引
chapters = []
for i, line in enumerate(lines):
    if chapter_pattern.match(line.strip()):
        chapters.append((len(chapters)+1, i, line.strip()))

# 加载记录
start_line = 0
if os.path.exists(record_file):
    with open(record_file, 'r') as f:
        try:
            start_line = int(f.read().strip())
            print(f"[记录模式] 从第 {start_line+1} 行继续阅读")
        except:
            pass

# 状态控制
current_mode = "manual"
delay = 2.0
stop_flag = False
current_line = start_line

def fullwidth_to_ascii(s: str) -> str:
    # 全角数字转半角
    fw_digits = "０１２３４５６７８９"
    ascii_digits = "0123456789"
    trans = {ord(f): a for f, a in zip(fw_digits, ascii_digits)}
    return s.translate(trans)

def chinese_to_int(ch: str) -> int:
    """
    将常见中文数字（如 "一二三", "十二", "一百二十三", "二零二五"）转换为整数。
    对非常规写法结果可能不完美，但覆盖常见小说章节表达。
    """
    ch = ch.strip()
    if not ch:
        return None

    # 先把全角数字转半角
    ch = fullwidth_to_ascii(ch)

    # 直接包含阿拉伯数字则直接解析
    m = re.search(r'\d+', ch)
    if m:
        return int(m.group())

    cn_digits = {'零':0,'〇':0,'○':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
    cn_units  = {'十':10,'百':100,'千':1000,'万':10000,'萬':10000}

    total = 0
    current = 0
    last_unit = 1
    i = 0
    while i < len(ch):
        c = ch[i]
        if c in cn_digits:
            current = current * 10 + cn_digits[c]
            i += 1
        elif c in cn_units:
            unit = cn_units[c]
            # 处理像 "十"、"十二" 的情况：若 current==0，视作 1*unit
            if current == 0:
                current = 1
            current *= unit
            total += current
            current = 0
            i += 1
        else:
            # 遇到非数字字符（标点/空格等）就跳过
            i += 1

    total += current
    return total if total != 0 else None

def show_chapter_list():
    print("\n📚 章节目录：")
    for num, idx, title in chapters:
        print(f"  {num:03d}: {title}")
    print("\n🔁 输入 'goto 章节号' 可跳转，如 `goto 5`，也支持 `goto 第十二章` 或 `goto 章节标题关键字`")

def input_listener():
    global current_mode, stop_flag, current_line, delay
    while not stop_flag:
        try:
            cmd = input().strip()
        except EOFError:
            # 在某些终端输入管道会触发 EOF，安全退出监听
            stop_flag = True
            break
        if not cmd:
            continue

        c = cmd.lower()
        if c == "a":
            current_mode = "auto"
            print("[切换为自动模式]")
        elif c == "m":
            current_mode = "manual"
            print("[切换为手动模式]")
        elif c == "q":
            stop_flag = True
            print("[退出阅读器，保存进度]")
        elif c == "chapters":
            show_chapter_list()
        elif c.startswith("goto"):
            # 解析参数部分（允许空格、中文标点等）
            arg = cmd.partition(" ")[2].strip()
            if not arg:
                print("[用法] 输入如 `goto 5` 或 `goto 第十二章` 或 `goto 章节标题关键字`")
                continue

            # 1) 尝试提取阿拉伯数字（含全角）
            arg_ascii = fullwidth_to_ascii(arg)
            m = re.search(r'(\d+)', arg_ascii)
            target_idx = None
            if m:
                target_idx = int(m.group(1))
            else:
                # 2) 尝试把中文数字（如 "第十二章" -> "十二"）解析为整数
                # 去掉常见前后缀
                tmp = re.sub(r'^[第\s]*', '', arg)
                tmp = re.sub(r'(章|回|节|卷|：|:|\.|,|，|\s)*$', '', tmp)
                parsed = chinese_to_int(tmp)
                if parsed is not None:
                    target_idx = parsed

            if target_idx is not None:
                # 按章节序号跳转（章节编号从 1 开始）
                chapter_match = [c for c in chapters if c[0] == target_idx]
                if chapter_match:
                    current_line = chapter_match[0][1]
                    print(f"[跳转到第 {target_idx} 章: {chapter_match[0][2]}]")
                else:
                    print("[错误] 没找到该章节编号")
            else:
                # 3) 当做标题关键词匹配（不区分大小写）
                kw = arg.lower()
                matched = [c for c in chapters if kw in c[2].lower()]
                if len(matched) == 0:
                    print("[错误] 没找到匹配的章节标题")
                elif len(matched) == 1:
                    current_line = matched[0][1]
                    print(f"[按标题匹配，跳转到: {matched[0][0]} - {matched[0][2]}]")
                else:
                    print("[注意] 找到多个匹配，列出前几个：")
                    for num, idx, title in matched[:10]:
                        print(f"  {num:03d}: {title}")
                    # 为了不阻断流程，自动跳到第一个匹配项
                    current_line = matched[0][1]
                    print(f"[已跳转到第一个匹配: {matched[0][0]} - {matched[0][2]}]")

        elif c == "1":
            print(f"[当前时间: {delay} 秒]")
            delay = max(0.1, delay - 0.1)
        elif c == "2":
            print(f"[当前时间: {delay} 秒]")
            delay = delay + 0.1
        elif c == "3":
            delay = 2.0
            print(f"[恢复默认时间: {delay} 秒]")
        else:
            print("[未知命令] 可用命令: a (自动), m (手动), q (退出), chapters (目录), goto ... (跳转), 1/2/3 (调整速度)")


listener_thread = threading.Thread(target=input_listener, daemon=True)
listener_thread.start()

# 阅读主循环
while current_line < len(lines):
    if stop_flag:
        break
    print(lines[current_line].strip())
    current_line += 1

    if current_line % 10 == 0:
        with open(record_file, "w") as f:
            f.write(str(current_line))

    if current_mode == "manual":
        input()
    else:
        time.sleep(delay)

# 结束保存
with open(record_file, "w") as f:
    f.write(str(current_line))
print(f"[已保存记录到第 {current_line} 行]")
