"""Release-tree static checks: bundled font, demo samples, OneID branding.

This does not start any process; it answers "does the checked-out tree actually
contain what the UI promises".

    python scripts/ci/e2e_static.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from e2e_common import ROOT, Reporter

APP = ROOT / "app" / "app.py"
FONT = ROOT / "tool_server" / "tool_workers" / "online_workers" / "fonts" / "NotoSansCJK-Regular.ttc"

HEADER_TITLE = "OneID : Remote Sensing Identity Interpretation"
HEADER_SUBTITLE = "Dual-source (RGB/SAR) shared-identity reasoning for remote sensing"
STALE_PATTERNS = (
    r"\bNPU\b",
    r"nwpu",
    r"Li Huihui",
    r"Intelligent Vision and Information Processing",
)


def main() -> int:
    report = Reporter("static release checks")

    # 1) bundled CJK font: present and able to draw Chinese
    if report.check("bundled font present", FONT.is_file(), str(FONT.relative_to(ROOT))):
        try:
            from PIL import Image, ImageDraw, ImageFont

            font = ImageFont.truetype(str(FONT), 32)
            canvas = Image.new("L", (260, 60), 0)
            ImageDraw.Draw(canvas).text((4, 8), "光学/SAR 目标", font=font, fill=255)
            ink = sum(1 for value in canvas.getdata() if value > 0)
            report.check("bundled font renders Chinese", ink > 200, f"ink={ink}px")
        except Exception as exc:  # pragma: no cover - environment dependent
            report.check("bundled font renders Chinese", False, repr(exc))

    # 2) every asset path referenced by the UI exists in the tree
    text = APP.read_text(encoding="utf-8")
    referenced = set(re.findall(r'["\']([^"\']*assets/[^"\']+)["\']', text))
    referenced |= {"." + m for m in re.findall(r'["\'](scripts/tools_test/[^"\']+\.(?:png|jpg))["\']', text)}
    missing = [path for path in sorted(referenced) if not (ROOT / path.lstrip("./")).is_file()]
    report.check("all referenced sample images exist", not missing,
                 ", ".join(missing) or f"{len(referenced)} paths")

    # 3) header branding
    report.check("header title is OneID", HEADER_TITLE in text)
    report.check("header subtitle is neutral", HEADER_SUBTITLE in text)
    hits = [p for p in STALE_PATTERNS if re.search(p, text, 0 if p == r"\bNPU\b" else re.I)]
    report.check("no leftover institutional branding", not hits, ", ".join(hits))
    report.check("logo is optional", 'env_path("LOGO_PATH", "")' in text)

    # 4) exactly one UI entry point
    apps = sorted(p.name for p in (ROOT / "app").glob("*.py"))
    report.check("single UI entry point", apps == ["app.py"], ", ".join(apps))

    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
