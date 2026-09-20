"""Private fresh-process worker. Reliability guards are NOT a sandbox.

The guard is adapted from Hugging Face evaluate's code_eval/execute.py:
Copyright 2020 The HuggingFace Datasets Authors and the current dataset script
contributor. Licensed under Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0
The full license is included in licenses/code_eval-APACHE-2.0.txt.
Provided AS IS, without warranties or conditions of any kind.

That guard is adapted from OpenAI human-eval, under the MIT License:
Copyright (c) OpenAI (https://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Modified: fresh exec process, OS-discarded output, resource limits, private
completion protocol and parent-managed timeout/cleanup replace multiprocessing.
"""

import builtins
import faulthandler
import importlib
import math
import os
import resource
import shutil
import subprocess
import sys


def reliability_guard():
    """Reduce accidental damage, without claiming to constrain hostile Python."""
    faulthandler.disable()
    for name in ("exit", "quit", "help"):
        setattr(builtins, name, None)
    for name in (
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
        "setsid",
        "setpgid",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("rmtree", "move", "chown"):
        setattr(shutil, name, None)
    subprocess.Popen = None
    for name in ("ipdb", "joblib", "psutil", "tkinter"):
        sys.modules[name] = None


def main():
    source, channel, timeout, memory, file_limit = sys.argv[1:]
    with open(source, encoding="utf-8") as handle:
        program = handle.read()
    channel = int(channel)
    os.set_inheritable(channel, False)
    # The HF evaluator initialized NumPy for its estimator before forking. Its
    # first import sets environment variables, which the accident guard blocks.
    # Initialize the required dependency before guarding or announcing readiness,
    # so an installation failure aborts evaluation instead of failing candidates.
    importlib.import_module("numpy")
    memory = int(memory)
    if memory:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    cpu = math.ceil(float(timeout)) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (int(file_limit), int(file_limit)))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    write, exit_process = os.write, os._exit
    reliability_guard()
    write(channel, b"R")
    try:
        # Execution is the metric's purpose, confined to this disposable worker.
        exec(compile(program, "<candidate>", "exec"), {})  # noqa: S102
    except BaseException:  # noqa: BLE001
        # Candidate SystemExit/KeyboardInterrupt also fail, rather than escape scoring.
        write(channel, b"F")
    else:
        write(channel, b"P")
    # Candidate-created atexit handlers and stream replacements cannot delay exit.
    exit_process(0)


if __name__ == "__main__":
    main()
