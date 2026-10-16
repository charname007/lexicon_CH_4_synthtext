"""临时：下载并验证 RRC MLT 标注包内容。用完即删。"""
import io
import zipfile
import requests

requests.packages.urllib3.disable_warnings()
HEADERS = {"User-Agent": "Mozilla/5.0"}

urls = [
    ("mlt19_train_gt",
     "https://rrc.cvc.uab.es/downloads/ch9_training_localization_transcription_gt.zip"),
    ("mlt17_train_gt",
     "https://rrc.cvc.uab.es/downloads/ch8_training_localization_transcription_gt_v2.zip"),
    ("mlt17_val_gt",
     "https://rrc.cvc.uab.es/downloads/ch8_validation_localization_transcription_gt_v2.zip"),
]
for tag, url in urls:
    r = requests.get(url, headers=HEADERS, timeout=60, verify=False)
    head = r.content[:8]
    print(f"=== {tag}: HTTP {r.status_code}, {len(r.content)} bytes, head={head[:4]!r}")
    if not r.content.startswith(b"PK"):
        print("  不是 zip！前 200 字节:", r.content[:200])
        continue
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    print(f"  成员数: {len(names)}, 示例: {names[:3]}")
    # 读第一个 txt 成员看内容
    txts = [n for n in names if n.lower().endswith(".txt")]
    if txts:
        sample = zf.read(txts[0]).decode("utf-8", "replace")
        print(f"  样本 {txts[0]}:\n{sample[:400]}")
