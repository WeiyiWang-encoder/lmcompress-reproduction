"""Insert the self-generated figures into the report and prepare it for Overleaf.

Three jobs:
  1. add \\includegraphics figure environments at the right places;
  2. make the preamble Overleaf-portable (engine hint, robust microtype, graphicspath);
  3. report anything that could behave differently on TeX Live than on MiKTeX.

No figure is copied from the paper: all three are plotted from this project's own
measurements by tools/make_report_figures.py.
"""

from __future__ import annotations

import io
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(HERE, "report", "report.tex")

# --------------------------------------------------------------------------- #
FIG_REPRODUCTION = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_reproduction.pdf}
\caption{Measured compression rate against the published figure, per modality,
on a logarithmic axis. The text and audio agreement is the reproduction's
positive result; the image and video bars are \emph{not} like-for-like, because
the official pipeline's numerator and denominator describe different objects
(Chapter~\ref{ch:audit}).}
\label{fig:reproduction}
\end{figure}
"""

FIG_METHODS = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_methods_per_modality.pdf}
\caption{Every measured method, by modality. Orange bars are the large-model
methods under reproduction; blue bars are the classical baselines measured
through the same harness, which serve as the control that validates the
apparatus. Note the image and video panels: the model bar is a rate for a
$32\times32$, 512-colour surrogate, not for the source data.}
\label{fig:methods}
\end{figure}
"""

FIG_ACCOUNTING = r"""
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/fig_image_accounting.pdf}
\caption{Where the image denominator comes from, and why the ratio is not a
source-image rate. The denominator counts full-resolution source channel samples
(\num{5870592} of \num{5872164}, the difference being the loader's floor
division); the numerator counts palette indices of a $32\times32$ quantised grid.
The two describe different objects.}
\label{fig:accounting}
\end{figure}
"""


def main() -> int:
    text = io.open(TEX, encoding="utf-8").read()
    original = text

    # ---- 1. engine hint for Overleaf -------------------------------------- #
    if text.startswith("%"):
        text = "% !TEX program = xelatex\n" + text
        print("  ok  added '% !TEX program = xelatex' engine hint")

    # ---- 2. graphicspath so figures resolve regardless of build directory -- #
    if "\\graphicspath" not in text:
        text = text.replace(
            "\\usepackage{graphicx}",
            "\\usepackage{graphicx}\n"
            "\\graphicspath{{figures/}{./}}", 1)
        print("  ok  added graphicspath")

    # ---- 3. microtype: keep protrusion, drop the engine-specific expansion -- #
    if "\\usepackage{microtype}" in text and "\\microtypesetup" not in text:
        text = text.replace(
            "\\usepackage{microtype}",
            "% microtype: protrusion is safe on every engine/font set; font expansion\n"
            "% requires scalable fonts and has caused engine-specific failures, so it is\n"
            "% disabled for portability.\n"
            "\\usepackage{microtype}\n"
            "\\microtypesetup{expansion=false}", 1)
        print("  ok  made microtype portable")

    # ---- 4. place the figures --------------------------------------------- #
    # fig:methods after the summary table in the Results chapter
    anchor1 = "\\subsection{Text}\n"
    if anchor1 in text and "fig:methods" not in text:
        text = text.replace(anchor1, FIG_METHODS + "\n" + anchor1, 1)
        print("  ok  inserted fig_methods_per_modality")

    # fig:accounting right after the reconciliation equation
    anchor2 = ("and $\\num{5872164} - \\num{5870592} = \\num{1572}$, the loader's floor-division\n"
               "loss.")
    if anchor2 in text and "fig:accounting" not in text:
        text = text.replace(anchor2, anchor2 + "\n" + FIG_ACCOUNTING, 1)
        print("  ok  inserted fig_image_accounting")
    else:
        # fall back: after the image accounting table
        anchor2b = "\\label{tab:imgaccount}\n\\end{table}"
        if anchor2b in text and "fig:accounting" not in text:
            text = text.replace(anchor2b, anchor2b + "\n" + FIG_ACCOUNTING, 1)
            print("  ok  inserted fig_image_accounting (fallback anchor)")

    # fig:reproduction at the head of the Results summary
    anchor3 = "\\subsection{Summary}\n"
    if anchor3 in text and "fig:reproduction" not in text:
        text = text.replace(anchor3, anchor3 + FIG_REPRODUCTION + "\n", 1)
        print("  ok  inserted fig_reproduction")

    if text != original:
        io.open(TEX, "w", encoding="utf-8", newline="\n").write(text)
        print("  -> wrote report.tex")
    else:
        print("  -- no changes made")
    return 0


if __name__ == "__main__":
    sys.exit(main())
