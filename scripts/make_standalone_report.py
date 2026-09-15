"""Inline a report's images and MathJax into one self-contained HTML file."""
from __future__ import annotations
import base64, re, sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_standalone.html")
root = src.parent
html = src.read_text()

# inline <img src="...png"> as data URIs
MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}
inlined = [0, 0]

def inline_img(m):
    path = root / m.group(1)
    if not path.exists():
        print("  MISSING:", path); return m.group(0)
    mime = MIME.get(path.suffix.lower())
    if mime is None:
        print("  skipped (unknown type):", path.name); return m.group(0)
    data = base64.b64encode(path.read_bytes()).decode()
    inlined[0] += 1
    inlined[1] += path.stat().st_size
    return f'src="data:{mime};base64,{data}"'

html = re.sub(r'src="([^"]+\.(?:png|jpe?g|gif|webp|svg))"', inline_img, html)
print(f"  inlined {inlined[0]} images ({inlined[1]/1024/1024:.1f} MB)")

leftover = re.findall(r'src="(?!data:)([^"]+)"', html)
if leftover:
    print("  WARNING, not inlined:", sorted(set(leftover)))

# inline the MathJax bundle so the file works offline
mj = root / "vendor/node_modules/mathjax/es5/tex-svg.js"
if mj.exists():
    html = html.replace(
        '<script defer src="vendor/node_modules/mathjax/es5/tex-svg.js"></script>',
        f"<script>{mj.read_text()}</script>")
    print(f"  inlined MathJax ({mj.stat().st_size/1024/1024:.1f} MB)")

dst.write_text(html)
print(f"\nwrote {dst}  ({dst.stat().st_size/1024/1024:.1f} MB, fully self-contained)")
