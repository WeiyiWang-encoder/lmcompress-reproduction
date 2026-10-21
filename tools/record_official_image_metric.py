"""Record the official image metric finding in the audit.

Established by measurement (see results/_official_quantity.txt and
results/_row_causality.txt):

  * tokenisation now matches exactly: 5733 units, 5870592 samples, identical ids;
  * on one chunk the official model invocation and our port assign IDENTICAL
    probabilities to IDENTICAL targets (1774.715 bits either way);
  * the official encoding loop is ``zip(raw_probs, sequence_array)``, so token t is
    coded with row t -- and row t is verified causal, i.e. computed from tokens
    [0..t], which includes token t itself;
  * consequently the official figure is 1.2839 bits/sample on chunk 0 against our
    port's 1.7348, and the official whole-image rate is 15.5144 % against our
    23.9257 % -- a factor of 1.542;
  * the official coder emits 165 bytes for chunk 0, confirming that the reported
    number is real emitted output rather than a modelled length.
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
AUDIT = os.path.join(HERE, "docs", "AUDIT_FOR_REVIEW.md")

SECTION = """### 4.4.5 The official image metric codes each token against a distribution that has seen it [M]

This is the reason our channel-corrected port still reports 23.93 % where the official
script reports 15.51 %, and it is a stronger finding than the denominator question.

**What the official loop does.** `language_model.compress` builds

```python
log_probs = predict_fn(sequence_array)          # (1024, 512)
raw_probs = log_probs.reshape(-1, log_probs.shape[-1])
...
for pdf, symbol in zip(raw_probs, sequence_array):
    encoder.encode(normalize_pdf_for_arithmetic_coding(pdf), symbol)
```

`predict_fn` runs the model on the **complete** token sequence and returns, at row `t`,
the softmax of the logits at position `t`. The zip then codes token `t` with row `t`.

**Why that matters.** We verified the model is causal, so row `t` is computed from
tokens `0..t` — which *includes token `t` itself*. Perturbing the token suffix from
position 501 onward leaves rows 0–500 bit-identical (`max |Δ logit| = 0`) while moving
rows 501+ by up to 33: the probe is live and the rows are causal
(`tools/test_official_row_causality.py`).

So at every position the distribution used to code a symbol has already conditioned on
that symbol. That is not a predictive distribution for the token being coded, and the
resulting bit count is **not a lossless code length for the sequence**: no decoder that
does not already know token `t` can reproduce it. Our port codes token `t` with the
distribution from position `t-1`, which is the conventional and decodable convention.

**Measured.** On one identical chunk (same ids, same model):

| quantity | value |
|---|---|
| official rows (row `t` for token `t`) | **1.2839** bits/sample |
| proper causal (row `t-1` for token `t`) | 1.7363 bits/sample |
| our port | **1.7348** bits/sample |
| official arithmetic coder, actually emitted | 165 bytes = 1.2891 bits/sample |

Our port agrees with the conventional causal figure to **0.09 %**. The official figure
is **1.35×** smaller, and its own coder really emits the smaller stream — so this is
realised output, not a modelling artefact.

**Whole-image consequence.** Official 15.5144 % against our channel-corrected 23.9257 %
— a factor of **1.542**.

**Reconciliation of the two effects.** Before this section the image discrepancy had two
candidate causes; both turn out to be real and they are independent:

| cause | direction | status |
|---|---|---|
| channel-wise vs whole-RGB tokenisation | our port was wrong; fixed in `uic.data.image_units` | **resolved** — the port now produces 5733 units, 5870592 samples and identical ids |
| row `t` used to code token `t` | official figure is optimistic relative to any decodable code | **open** — and not something our port can or should reproduce |

Consequently the published image rates should be read as the size of a stream produced
under a convention whose decodability we cannot reconstruct. Our port's figure is the
conventional one. **Neither number is a like-for-like comparison with the classical
baselines**, which remain valid.

**Note on the same pattern elsewhere.** The audio path was checked for exactly this and
did **not** show it: the reference audio routine scores each byte against the
distribution its own context produced, and a future-byte perturbation leaves the score
unchanged (`max |Δ log p| = 0`). The two modalities therefore differ, and the audio
result is unaffected by this section.
"""


def main() -> int:
    text = io.open(AUDIT, encoding="utf-8").read()
    if "4.4.5" in text or "has seen it" in text:
        print("  -- already recorded")
        return 0
    anchor = "### 4.5 Video"
    if anchor not in text:
        print("  !! anchor not found")
        return 1
    text = text.replace(anchor, SECTION + "\n" + anchor, 1)
    io.open(AUDIT, "w", encoding="utf-8", newline="\n").write(text)
    print("  ok  added section 4.4.5 to AUDIT_FOR_REVIEW.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
