"""Build docs/Room-Scanner_submission.pdf (summary + technical report + benchmark) for the submission form.

    .venv/bin/python scripts/build_submission_pdf.py
Markdown -> HTML (python-markdown) -> PDF (headless Chrome). Images are embedded from docs/img.
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CHROME = next((p for p in ["/Users/manojmittal/Desktop/Google Chrome.app/Contents/MacOS/Google Chrome",
                           "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"] if Path(p).exists()), None)
parts = [("Summary", "docs/SUBMISSION.md"), ("Technical report", "docs/TECH_REPORT.md"), ("Benchmark", "docs/BENCHMARK.md")]
figs = [("Photo tier, house_b (default matcher): 3 rooms linked; 3 measured but not linked, drawn dashed", "docs/img/final_photo_plan.png"),
        ("Photo tier with MASt3R matches on a GPU (E32): 5 of 7 rooms linked, layout unverified",
         "docs/img/e32_mast3r_pipeline_plan.png"),
        ("Why the whole-house video fails (E34): walls duplicated across 17 COLMAP pieces", "docs/img/e34_video_topdown.png")]
html = ["<html><head><meta charset='utf-8'><style>",
        "body{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;max-width:180mm;margin:auto}",
        "h1{font-size:17pt;border-bottom:2px solid #333}h2{font-size:13pt;margin-top:1.2em}h3{font-size:11pt}",
        "table{border-collapse:collapse;font-size:9pt;margin:.5em 0}td,th{border:1px solid #bbb;padding:3px 5px}",
        "code{font-size:9pt;background:#f2f2f2}pre{font-size:8.5pt;background:#f6f6f6;padding:6px;white-space:pre-wrap}",
        ".pb{page-break-before:always}img{max-width:100%;border:1px solid #ddd}figcaption{font-size:9pt;color:#555}",
        "</style></head><body>"]
for i, (_, path) in enumerate(parts):
    md = (ROOT / path).read_text()
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", md)  # repo-relative links don't resolve in a PDF
    body = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html.append(f"<div class='{'pb' if i else ''}'>{body}</div>")
html.append("<div class='pb'><h1>Figures</h1>")
for cap, img in figs:
    b64 = base64.b64encode((ROOT / img).read_bytes()).decode()
    html.append(f"<figure><img src='data:image/png;base64,{b64}'><figcaption>{cap}</figcaption></figure>")
html.append("</div></body></html>")
src = ROOT / "data/derived/submission.html"
src.parent.mkdir(parents=True, exist_ok=True)
src.write_text("\n".join(html))
out = ROOT / "docs/Room-Scanner_submission.pdf"
if not CHROME:
    sys.exit("Chrome not found")
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                f"file://{src}"], check=True, capture_output=True)
print(out, f"{out.stat().st_size / 1e6:.2f} MB")
