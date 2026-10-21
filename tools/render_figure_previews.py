"""Re-render the report figures as PNG, purely for visual inspection.

The PDF versions are what LaTeX includes; PNGs exist so a human (or an agent) can
look at the result without a PDF rasteriser.
"""

from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(HERE, "tools", "make_report_figures.py"), encoding="utf-8").read()
# swap the output extension everywhere and execute in this namespace
src = src.replace('.pdf"', '.png"')
ns = {"__name__": "__main__", "__file__": os.path.join(HERE, "tools", "make_report_figures.py")}
print("rendering PNG copies ...")
exec(compile(src, "make_report_figures.py", "exec"), ns)  # noqa: S102
