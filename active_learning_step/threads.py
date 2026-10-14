# -*- coding: utf-8 -*-

"""How many CPU threads torch uses: no scheduler-specific code.

The thread count is, in order:

1. the one asked for (``--threads``);
2. ``OMP_NUM_THREADS``, which SEAMM (``seamm_exec``) sets for the codes it runs from
   the job's allocation, as do PBS and most launchers;
3. the CPUs this process may run on (its affinity mask), which Slurm and PBS cpusets,
   ``taskset`` and containers restrict; on a plain machine, all of them.

So it works the same under Slurm, PBS, SEAMM's own JobServer or no scheduler at all,
and never uses more cores than the process was given.
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["resolve_threads", "set_threads"]


def _affinity():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # macOS and others: no affinity mask
        return os.cpu_count() or 1


def resolve_threads(threads=None, environ=None, affinity=None):
    """The number of threads to use, and where it came from.

    Parameters
    ----------
    threads : int, optional
        The number asked for.
    environ : mapping, optional
        The environment, by default ``os.environ``.
    affinity : int, optional
        The CPUs available, by default from the process's affinity mask.

    Returns
    -------
    (int, str)
        The thread count and its source: ``"requested"``, ``"OMP_NUM_THREADS"``
        or ``"affinity"``.
    """
    if threads is not None:
        if int(threads) < 1:
            raise ValueError(f"threads must be at least 1, not {threads}")
        return int(threads), "requested"
    environ = os.environ if environ is None else environ
    value = environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    if value:
        try:
            n = int(value)
        except ValueError:
            n = 0
        if n >= 1:
            return n, "OMP_NUM_THREADS"
    n = _affinity() if affinity is None else int(affinity)
    return max(1, n), "affinity"


def set_threads(threads=None):
    """Set torch's intra-op thread count by :func:`resolve_threads`; return it."""
    import torch

    n, source = resolve_threads(threads)
    torch.set_num_threads(n)
    logger.info(f"torch uses {n} CPU threads ({source})")
    return n
