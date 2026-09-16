"""Reliable process exit after persisted results on this Isaac Sim / Kit installation.

Kit extension teardown hangs or segfaults on this host. These standalone commands
own their process; after writers are closed, OS process cleanup releases CUDA and
Vulkan resources. Exit codes must preserve failures (Kit fast shutdown uses 0).
"""
import os
import sys


def finish(exit_code: int = 0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
