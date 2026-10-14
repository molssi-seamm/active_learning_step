# -*- coding: utf-8 -*-

"""Base class for substeps in the Active Learning step."""

import logging
from pathlib import Path
import shlex
import shutil

import seamm
import seamm_exec
import seamm_util.printing as printing
from seamm_util.printing import FormattedText as __

import active_learning_step
from .check import member_dirs

logger = logging.getLogger(__name__)
job = printing.getPrinter()
printer = printing.getPrinter("Active Learning")


class Substep(seamm.Node):
    """A base class for substeps in the Active Learning subflowchart.

    The top-level ActiveLearning step iterates over the substeps
    of its subflowchart and calls each substep's `run()` method. Each substep
    is responsible for preparing its own input, invoking Active Learning,
    analyzing its output, and storing its results.

    This base class provides:

    * `is_runable` - whether the top-level loop should call `run()` on this
      substep. Override to return False for substeps that only contribute
      input (e.g., an Initialization step) rather than invoking the code.
    * `input_only` - a flag for "write input but don't run" mode, useful for
      debugging and testing.
    * `options` / `global_options` - delegated to the parent step so that
      command-line options reach substeps uniformly.
    * `header` - a printable header for this section of step.out.
    * `version` / `git_revision` - delegated to the plug-in package so that
      every substep reports the plug-in's version, not seamm.Node's.
    """

    #: The state of each job, by its root directory (see :attr:`state`)
    _state = {}

    def __init__(
        self,
        flowchart=None,
        title="no title",
        extension=None,
        logger=logger,
        module=None,
    ):
        self._input_only = False

        super().__init__(
            flowchart=flowchart,
            title=title,
            extension=extension,
            logger=logger,
            module=module,
        )

    @property
    def header(self):
        """A printable header for this section of output."""
        return "Step {}: {}".format(".".join(str(e) for e in self._id), self.title)

    @property
    def version(self):
        """The semantic version of the active_learning_step package."""
        return active_learning_step.__version__

    @property
    def git_revision(self):
        """The git revision of the active_learning_step package."""
        return active_learning_step.__git_revision__

    @property
    def global_options(self):
        """The global SEAMM options, delegated to the parent step."""
        return self.parent.global_options

    @property
    def options(self):
        """The Active Learning options, delegated to the parent step."""
        return self.parent.options

    @property
    def input_only(self):
        """Whether to write the input only, not run Active Learning."""
        return self._input_only

    @input_only.setter
    def input_only(self, value):
        self._input_only = value

    @property
    def is_runable(self):
        """Whether this substep should be run by the top-level loop.

        Override to return False for substeps that only contribute input
        (e.g., an Initialization step) rather than invoking
        Active Learning themselves.
        """
        return True

    # ------------------------------------------------------------------
    # What the sub-steps of a job share: the last committee, scores, ...
    # ------------------------------------------------------------------

    @property
    def state(self):
        """The active-learning results of this job so far, shared by the sub-steps.

        ``committee`` is the last committee trained or used, ``scores`` the last
        scores files and ``pool`` how they were made (``"database"`` or
        ``"file"``). A later Active Learning step in the same job sees what an
        earlier one left, so a flowchart can train a committee, run molecular
        dynamics with it, then score and select.
        """
        root = str(self.flowchart.root_directory or "")
        return Substep._state.setdefault(root, {})

    def committee_directory(self, P, key="committee"):
        """The committee to use: the one trained or used earlier, or a directory.

        Parameters
        ----------
        P : dict
            The step's parameters.
        key : str
            The parameter naming the committee.

        Returns
        -------
        pathlib.Path
        """
        value = str(P[key]).strip()
        if value in ("previous", "the previous committee"):
            committee = self.state.get("committee")
            if committee is None:
                raise RuntimeError(
                    f"{self.title}: there is no previous committee in this job. Train "
                    "one with Train Committee first, or give its directory."
                )
            return Path(committee)
        path = self.file_path(value, read_only=True)
        if not path.is_dir():
            raise FileNotFoundError(f"{self.title}: there is no committee {path}")
        if not member_dirs(path):
            raise FileNotFoundError(
                f"{self.title}: {path} has no members (directories m0, m1, ...)"
            )
        return path

    def resolve_files(self, patterns, what="file"):
        """The files matching whitespace-separated names or glob patterns.

        Relative names are taken from the job's directory; ``job:`` references and
        absolute paths work as in other steps.
        """
        if isinstance(patterns, (list, tuple)):
            items = [str(p) for p in patterns]
        else:
            items = shlex.split(str(patterns))
        root = Path(self.flowchart.root_directory)
        files = []
        for item in items:
            if any(c in item for c in "*?["):
                base = self.file_path(".", relative_to=root, read_only=True)
                path = Path(item).expanduser()
                if path.is_absolute():
                    matches = sorted(Path(path.anchor).glob(str(path)[1:]))
                else:
                    matches = sorted(base.glob(item))
                if not matches:
                    raise FileNotFoundError(f"{self.title}: no {what} matches {item}")
                files.extend(matches)
            else:
                path = self.file_path(item, relative_to=root, read_only=True)
                if not path.exists():
                    raise FileNotFoundError(f"{self.title}: there is no {what} {path}")
                files.append(path)
        if not files:
            raise ValueError(f"{self.title}: no {what}s were given")
        return files

    # ------------------------------------------------------------------
    # Running seamm-al in the MLFF environment
    # ------------------------------------------------------------------

    def mlff_config(self):
        """How to run code in the MLFF environment: the xnn plug-in's xnn.ini.

        The committee needs torch and xnn, which live in the environment the xnn
        plug-in runs its models in, so this step uses the same one.
        """
        from xnn_step import XnnStep

        executor = self.parent.flowchart.executor
        return XnnStep.get_executor_config(executor, self.global_options)

    def device(self, P):
        """The torch device: the step's, else the one in xnn.ini, else cpu."""
        device = str(P.get("device", "default")).strip()
        if device in ("", "default"):
            device = str(self.mlff_config().get("device", "cpu")).strip() or "cpu"
        return device

    def run_seamm_al(self, args):
        """Run ``seamm-al`` in the MLFF environment, in this step's directory.

        This package's own modules are copied into ``_code/`` and put on the
        ``PYTHONPATH``, so the code that runs is the code of this SEAMM installation
        and nothing has to be installed in the MLFF environment. They need only
        numpy, ase, torch and xnn there; the step's own classes, which need SEAMM,
        are imported only where SEAMM is.

        Parameters
        ----------
        args : [str]
            The arguments of ``seamm-al``, e.g. ``["score", "pool.extxyz"]``.

        Returns
        -------
        seamm_exec.TaskResult
            The result; ``None`` when only writing the input. The output is in
            ``stdout.txt`` and ``stderr.txt``, and everything ``seamm-al`` writes
            stays in the step's directory.
        """
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        code = directory / "_code" / "active_learning_step"
        if code.exists():
            shutil.rmtree(code)
        package = Path(active_learning_step.__file__).parent
        shutil.copytree(
            package,
            code,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        config = dict(self.mlff_config())
        installation = config.get("installation", "conda")
        if installation == "local":
            # 'code' is the xnn command; python is beside it
            xnn = Path(shlex.split(config.get("code", "xnn"))[0])
            python = str(xnn.with_name("python")) if xnn.parent != Path() else "python"
        else:
            python = "python"
        cmd = [python, "-m", "active_learning_step.cli", *(str(a) for a in args)]
        command = " ".join(shlex.quote(word) for word in cmd)
        self.logger.info(f"Running seamm-al: {command}")
        root = str(self.flowchart.root_directory or "")
        shown = " ".join(_relative(word, root) for word in cmd[3:])
        printer.normal(__(f"    seamm-al {shown}", wrap=False, dedent=False))
        printer.normal("")

        if self.input_only:
            (directory / "command.txt").write_text(command + "\n")
            return None

        # torch's threads: the cores of the job, or fewer if SEAMM is told so
        ce = seamm_exec.computational_environment()
        # one process: the cores of this node (a scheduler's tasks x their CPUs)
        per_node = ce.get("NTASKS_PER_NODE", ce.get("NTASKS", 1))
        n_cores = int(per_node or 1) * int(ce.get("CPUS_PER_TASK", 1) or 1)
        ncores = self.global_options.get("ncores", "available")
        if ncores != "available":
            n_cores = min(n_cores, int(ncores))
        n_cores = max(1, n_cores)
        env = {
            "PYTHONPATH": str(code.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": str(n_cores),
        }

        task = seamm_exec.Task(
            key="seamm-al",
            program="xnn",
            # braces would be taken for the back end's {code}, {NTASKS}, ...
            cmd=[shlex.quote(w).replace("{", "{{").replace("}", "}}") for w in cmd]
            + [">", "stdout.txt", "2>", "stderr.txt"],
            config=config,
            directory=self.directory,
            in_situ=True,
            shell=True,
            env=env,
            keep=["."],
            resources=seamm_exec.Resources(ntasks=1, cpus_per_task=n_cores),
        )
        result = seamm_exec.run_task(
            task, node=self, executor=self.parent.flowchart.executor
        )
        if not result.ok:
            stderr = directory / "stderr.txt"
            text = stderr.read_text() if stderr.exists() else result.stderr
            tail = "\n".join(text.splitlines()[-20:])
            raise RuntimeError(
                f"{self.title}: seamm-al failed ({result.reason}):\n{tail}"
            )
        return result


def _relative(word, root):
    """A path in the job shown relative to the job's directory, for printing."""
    if root and word.startswith(root.rstrip("/") + "/"):
        return word[len(root.rstrip("/")) + 1 :]
    return word
