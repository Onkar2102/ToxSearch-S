"""
Publication-safe matplotlib PDF/PS output: embed outline fonts, avoid Type 3 text.

Matplotlib's PDF backend can emit text as Type 3 (bitmap) glyphs. Setting
``pdf.fonttype`` / ``ps.fonttype`` to **42** embeds TrueType fonts as PDF
Type 42 streams (subset embedding), which satisfies venues that require
Type 1 or TrueType with fonts embedded (e.g. ACM).

This does **not** switch to PostScript Type 1 fonts; it uses the standard
Matplotlib fix for vector text in PDF.

Call ``configure_matplotlib_embedded_fonts()`` once per process after
``import matplotlib`` and before ``savefig`` to PDF/PS.
"""


def configure_matplotlib_embedded_fonts() -> None:
    import matplotlib

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
