import os
import re

ROWS = [("v0", "article", 11, 1.0), ("v1", "article", 12, 1.5),
        ("v2", "article", 12, 1.8), ("v3", "extarticle", 13, 1.5),
        ("v4", "extarticle", 13, 1.9), ("v5", "extarticle", 14, 1.4),
        ("v6", "extarticle", 14, 1.6), ("v7", "extarticle", 15, 1.4)]

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(os.path.dirname(HERE), "report")

print("  %-4s %-11s %5s %8s %8s %8s" % ("var", "cls", "pt", "stretch", "ch2", "total"))
print("  " + "-" * 50)
for n, c, s, st in ROWS:
    p = os.path.join(REPORT, "_" + n + ".aux")
    if not os.path.exists(p):
        print("  %-4s  (no aux)" % n)
        continue
    a = open(p, encoding="utf-8", errors="replace").read()
    d = dict((m.group(1), m.group(2)) for m in
             re.finditer(r"newlabel\{([^}]*)\}\{\{[^}]*\}\{(\d+)\}", a))
    m = re.search(r"lastpage@lastpage\{(\d+)\}", a)
    p1 = int(d.get("ch:lit-start", -1))
    p2 = int(d.get("ch:lit-end", -1))
    pg = p2 - p1 + 1 if p1 > 0 and p2 > 0 else -1
    flag = "   <== 7-8 pages" if 7 <= pg <= 8 else ""
    print("  %-4s %-11s %5d %8.1f %8s %8s%s"
          % (n, c, s, st, pg, m.group(1) if m else "?", flag))
