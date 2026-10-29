# -*- coding: utf-8 -*-

"""Writing extended XYZ files with ASE so that they read back as written.

ASE quotes a value on the comment line only if it holds a space, a quote, a brace or
a bracket. A value with '=' but no space -- a configuration name such as
'LiDMCct_chel_ctmin_toO=' -- is written bare, and ASE then reads
'name=LiDMCct_chel_ctmin_toO= pbc="F F F"' as the one value
'LiDMCct_chel_ctmin_toO=pbc=F F F' (ASE 3.29). Writing inside
:func:`quoted_extxyz` quotes such values, and backslashes, too.
"""

import contextlib


@contextlib.contextmanager
def quoted_extxyz():
    """Within this context, ASE's extended XYZ writer also quotes values that
    contain '=' or a backslash, escaping backslashes and double quotes."""
    import ase.io.extxyz as extxyz

    original = extxyz.escape

    def escape(string):
        if "=" in string or "\\" in string:
            string = string.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{string}"'
        return original(string)

    extxyz.escape = escape
    try:
        yield
    finally:
        extxyz.escape = original
