"""Render Architecture IR to HTML / Jupyter."""

from .html import render_html, write_html
from .jupyter import show

__all__ = ["render_html", "write_html", "show"]
