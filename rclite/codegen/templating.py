"""Shared Jinja2 renderer for target *scaffolding* templates.

Layer-1 "scaffolding" — Arduino sketches, the Cortex-M0/GBA/NES ``main.c``
harnesses and the WASM Rust drivers — is mostly-static text with a handful of
injected values. Historically each target hand-chained
``str.replace("@@X@@", ...)`` calls; this module centralises that into a single
Jinja2 environment.

The placeholder delimiters stay as ``@@VAR@@`` (the pre-existing convention),
so the template files are unchanged and the rendered output is **byte-identical**
to the old replace-chains. The only behavioural change is robustness:
``StrictUndefined`` turns a forgotten context value into an immediate
``UndefinedError`` instead of a silently-unfilled ``@@X@@`` placeholder.

This deliberately covers only the *scaffolding* layer. The arithmetic kernel
emitters (``c_kernel_symmetric``, ``arduino/emit_c``, the affine/sparse
lowerers) stay as Python — their value is bit-exact codegen logic, not text
assembly, and that logic belongs next to the constants it computes.
"""

from __future__ import annotations

import pathlib
from functools import lru_cache

import jinja2


@lru_cache(maxsize=None)
def _env_for(root: str) -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(root),
        variable_start_string="@@",
        variable_end_string="@@",
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render_template(path, /, **context) -> str:
    """Render the scaffolding template at *path* with *context* values.

    ``@@NAME@@`` placeholders are substituted from *context*; a placeholder with
    no matching key raises :class:`jinja2.UndefinedError` (the old ``.replace``
    chain silently left it unfilled).
    """
    path = pathlib.Path(path)
    return _env_for(str(path.parent)).get_template(path.name).render(**context)
