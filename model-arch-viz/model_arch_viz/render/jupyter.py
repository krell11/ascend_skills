"""Jupyter display helpers."""

from __future__ import annotations

from ..ir import Architecture
from .html import render_html


def show(arch: Architecture):
    """Display the architecture widget in a Jupyter notebook."""
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            "IPython is required for show(). Install with: pip install ipython"
        ) from exc
    return display(HTML(render_html(arch, standalone=False)))


def html(arch: Architecture) -> str:
    """Return the widget HTML fragment (for custom notebook embedding)."""
    return render_html(arch, standalone=False)
