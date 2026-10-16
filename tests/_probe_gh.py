"""临时：批量检查 GitHub 数据集仓库的标注获取方式（README 下载链接 + 仓库文件树）。用完即删。"""
import base64
import re
import requests

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}

repos = [
    "HCIILAB/TKH_MTH_Datasets_Release",
    "HCIILAB/M5HisDoc",
    "Pengjie-W/HUST-OBC",
    "QwenLM/CC-OCR",
    "detectRecog/CCPD",
]

LINK_RE = re.compile(
    r"https?://[^\s)\"']+",
)

for repo in repos:
    print(f"\n===== {repo} =====")
    # README
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/readme",
                         headers=HEADERS, timeout=30)
        if r.status_code == 200:
            content = base64.b64decode(r.json()["content"]).decode("utf-8", "replace")
            links = []
            for m in LINK_RE.findall(content):
                if any(k in m.lower() for k in
                       ["zip", "tar", "drive.google", "pan.baidu", "1drv",
                        "onedrive", "release", "download", "huggingface"]):
                    links.append(m)
            for l in sorted(set(links))[:15]:
                print("  link:", l)
        else:
            print("  README HTTP", r.status_code)
    except Exception as e:
        print("  README ERR", e)

    # 顶层文件树（看标注是否直接在仓库里）
    try:
        r = requests.get(
            f"https://api.github.com/repos/{repo}/git/trees/HEAD",
            headers=HEADERS, timeout=30)
        if r.status_code == 200:
            paths = [t["path"] for t in r.json().get("tree", [])]
            ann = [p for p in paths if re.search(
                r"(gt|label|annot|transcri|char|dict|\.txt$|\.json$|\.csv$)",
                p, re.I)]
            print("  tree:", ", ".join(paths[:20]))
            if ann:
                print("  标注样文件:", ann[:10])
    except Exception as e:
        print("  tree ERR", e)
