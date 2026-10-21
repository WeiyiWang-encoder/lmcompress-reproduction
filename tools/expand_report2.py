"""Add the two remaining substantive chapters to reach 30 pages.

  * an experiment-design chapter covering the outstanding paired-context run and
    the alternative-model feasibility study (the task-3 material), including a
    runnable protocol;
  * a chapter on the independent-review protocol used in this project, which is a
    genuine part of the methodology rather than filler.
"""

from __future__ import annotations

import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(HERE, "report", "report.tex")

EXPERIMENTS = r"""
% ===========================================================================
\section{Designed experiments and their feasibility}
\label{ch:experiments}
% ===========================================================================

Chapter~\ref{ch:budget} identified one unresolved effect of the same magnitude as
the variable that confounds it. This chapter specifies the experiment that would
settle it, and the feasibility of the wider study the capstone brief invites.

\subsection{The paired-context experiment}

\textbf{Question.} Of the measured $\num{+0.312}$\,pp penalty from the published
LoRA adapter, how much is attributable to the adapter and how much to the shorter
context window?

\textbf{Design.} A $2\times2$ factorial over \{base, $+$LoRA\} $\times$ \{4096,
8192\}, on the same three MeDAL files, with the adapter evaluated \emph{unmerged}
and the base loaded in the published 8-bit configuration. The primary outcome is
the compression rate; the quantity of interest is the interaction term,
$\Delta_{8192} - \Delta_{4096}$, which isolates the adapter effect from the
context effect.

\textbf{Why unmerged.} Section~\ref{ch:finetune} showed that merging contributes
\num{2.84}\,\% of the measured penalty on an unquantised base, and can in some
inputs round the adapter away entirely. An unmerged evaluation removes the
question.

\textbf{Resources.} Two arms at context \num{8192} over three files. At the
measured local throughput for the published 8-bit configuration
($\approx\num{2744.77}$\,s for the whole MeDAL test set in the paper's own
timing, scaled to our three files), this is a matter of hours on a single
accelerator and is the recommended first use of the accompanying notebook.

\textbf{Decision rule.} If the interaction is small relative to the main effect of
the adapter, the penalty is attributable to the adapter and the published
\SI{0.2}{\percent} gain is not reproducible with the released checkpoint. If the
interaction is comparable to the penalty, the effect is a context-length artefact
and the published direction may still hold at the paper's own setting. Both
outcomes are informative; the current state of the evidence supports the former
but does not establish it.

\subsection{The training-quantity ablation the paper omits}

The paper reports 10 epochs for text fine-tuning, and its own released checkpoint
shows evaluation loss rising from step \num{1020} onward. It reports no ablation
of the training budget. A sweep over $\{1,2,5,10\}$ epochs, on a domain where
fine-tuning demonstrably helps, would determine whether the MeDAL null result is a
training-quantity effect.

The appropriate domain is the Pile of Law \code{eurlex} split, where the published
gain is \SI{16}{\percent} relative (\num{5.86} $\to$ \num{4.90}) rather than
\num{0.2}{\percent}. A monotone decrease in rate with epochs would support a
quantity explanation; a minimum near two epochs followed by regression would
confirm the over-training mechanism measured in Chapter~\ref{ch:finetune}. The
same run should checkpoint per epoch, which permits the ablation to be read off a
single training job rather than four.

Cost: the published recipe fine-tunes on \SI{64}{MB} of text for 10 epochs. That is
the dominant cost of the whole study and is the reason this experiment is
specified for a multi-hour accelerator session rather than the local workstation.

\subsection{Alternative predictors and the tokeniser confound}

The adaptation claim need not be tested with LoRA on an 8-billion-parameter model.
Two smaller options are worth specifying, because they are cheap and they control
different things.

\begin{table}[h]
\centering
\small
\caption{Fine-tuning options, and what each controls for.}
\label{tab:models2}
\begin{tabularx}{\textwidth}{L{2.9cm}rrL{4.4cm}}
\toprule
Model & Params & Local feasibility & What it controls for \\
\midrule
GPT-2 & 124\,M & fine: $\approx$2\,GB, $\approx$0.5--1\,h per epoch on
  \SI{64}{MB} & Whether any causal LM adapts to the domain; cheapest meaningful
  control \\
ByT5-base & 582\,M & tight: $\approx$7\,GB, needs short context &
  Byte-level operation, removing tokeniser confounds entirely; its decoder can be
  used causally \\
Llama-3-8B $\pm$ LoRA & 8\,B & no: 4-bit path currently non-functional locally &
  The published configuration itself \\
\bottomrule
\end{tabularx}
\end{table}

\textbf{Two cautions.} First, a masked language model is unsuitable for this
comparison: the method requires a causal next-token distribution, and BERT-style
objectives do not provide one. ByT5 and GPT-2 are appropriate; BERT is not.
Second, changing the model changes the tokeniser, and a rate comparison across
models therefore mixes two effects. The clean framing is to hold the corpus fixed
and report \textbf{bits per source byte}, which is tokeniser-independent --- the
unit this report already uses throughout.

\textbf{The binding constraint is time, not memory.} A \SI{64}{MB} corpus is of
order $16$ million tokens. At the measured local throughput for a
billion-parameter model, a single epoch is measured in days, which is why any
faithful reproduction of the fine-tuning claim belongs on rented accelerator time
regardless of which model is chosen.

\subsection{Closing the image and video distribution gap}

The image baseline comparison in Section~\ref{sec:imgbase} is not
distributionally matched: our single aerial-detail image compresses to
\SI{84.4}{\percent} under PNG against a published \SI{47.1}{\percent}. Repeating
the baselines on CLIC2019 Professional and ILSVRC would make the magnitudes
comparable. Note that this affects only the baseline magnitudes: the
non-injectivity finding of Chapter~\ref{ch:audit} is a property of the processor
and is independent of which image is used, and the denominator convention is
likewise image-independent.
"""

REVIEW_PROTOCOL = r"""
% ===========================================================================
\section{Independent-review protocol}
\label{ch:review}
% ===========================================================================

Because this reproduction makes claims critical of a published evaluation, and
because three of its own claims were falsified in the course of the work, the
project adopted an explicit adversarial review protocol. This chapter documents
it, both as a record of method and as a template for similar work.

\subsection{Why adversarial review, and not self-review}

The failure mode the project repeatedly exhibited was not faulty reasoning but
\emph{unexamined assumptions about where evidence lives}: a dependency search that
excluded the virtual environment, an index derivation that assumed a sixteen-slot
row where the grid has seventeen, a pipeline of our own mistaken for the pipeline
under audit. In each case the reasoning downstream of the assumption was sound,
and re-reading it would not have revealed anything. What revealed the errors was
running something that could contradict them.

\subsection{The review guide}

\file{docs/REVIEW\_GUIDE\_FOR\_AGENT.md} is written for an autonomous reviewer
whose instruction is to falsify rather than to agree. It contains:

\begin{enumerate}[leftmargin=1.4em,itemsep=2pt]
\item \textbf{Ground rules.} The reviewer is told there are two codebases --- the
      authors' reference implementation and our clean-room port --- and that a
      defect in the first is a finding while a defect in the second is a bug. Both
      are explicitly stated to be possibly wrong.
\item \textbf{A claim table} with confidence levels and severity, including
      retracted claims marked as such.
\item \textbf{An explicit ``what is not claimed'' section}, so that the reviewer
      can assess whether the report's tone matches its evidence. It states that we
      do not claim fabrication, that we do not claim the audio headline figure is
      wrong, that we do not claim the image or video denominators are
      miscalculated, and that we do not claim the models are incapable.
\item \textbf{Per-claim verification and falsification instructions.} Each claim
      is paired with the command that re-derives it and an explicit statement of
      what result would destroy it.
\item \textbf{A required reporting format} that forces the reviewer to state the
      strongest counter-argument and the residual uncertainty for every claim, not
      merely agreement or disagreement.
\item \textbf{A section listing what would overturn each conclusion}, on the
      principle that a claim which cannot be overturned is not a finding.
\end{enumerate}

\subsection{The executable gates}

Two scripts turn the protocol into something that can be run rather than merely
read.

\paragraph{\file{tools/review\_harness.py}.} Ten checks covering the text
alignment guard, the withdrawn audio claim (as a regression guard on the
documentation rather than on the code), image non-injectivity, the loader loss,
the reference log's staleness, the LoRA comparison, the absence of
\code{bits2bytes}, the coder guard, the reported-rate convention, and the video
baselines. Design rules: every check prints the raw quantities it measured rather
than only a verdict; a check that raises is reported as a failure rather than
skipped silently; and expected values are stated only where the expectation is
itself the claim.

\paragraph{\file{tools/check\_doc\_consistency.py}.} Encodes the canonical facts
once and scans all project documents for superseded values outside a labelled
retraction, and for required values that have gone missing. Its exit code
functions as a commit gate. This exists because the project's dominant error mode
was cross-document residue: a number corrected in one place while another
document still asserted the old value, which human re-reading did not reliably
catch.

\subsection{Retention of retractions}

Every withdrawn claim is retained in this report and in the project's audit
artefact, with its root cause. The reasoning is not sentimental. First, a reader
who cannot see what was withdrawn cannot judge how much weight the surviving
claims bear. Second, several retractions document error modes that a reader is
likely to share --- notably the temptation to attribute a missing artefact to the
artefact's absence. Third, the retractions are themselves evidence about the
limits of unreviewed self-verification, which is a finding of the project.

\subsection{Limitations of the protocol}

The reviewer in this project was a single automated agent, and the review guide
was written by the same author as the claims. Both facts limit what the protocol
can establish. An adversarial reviewer that shares the author's priors may fail to
question the same assumptions; the two errors that were caught by tooling were
caught because the tooling encoded arithmetic independent of the prose, not
because the reviewer disagreed. Genuinely independent review --- by a person who
did not write the harness --- remains outstanding and is recommended before any of
these findings are submitted for publication.
"""


def main() -> int:
    text = io.open(TEX, encoding="utf-8").read()

    a = "% ===========================================================================\n\\section{Retractions and corrections}"
    if a in text:
        text = text.replace(a, EXPERIMENTS + "\n" + a, 1)
        print("  ok  experiments chapter inserted")
    else:
        print("  !! retractions anchor missing")

    b = "% ===========================================================================\n\\section{Discussion}"
    if b in text:
        text = text.replace(b, REVIEW_PROTOCOL + "\n" + b, 1)
        print("  ok  review-protocol chapter inserted")
    else:
        print("  !! discussion anchor missing")

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(text)
    print("  -> wrote report.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
