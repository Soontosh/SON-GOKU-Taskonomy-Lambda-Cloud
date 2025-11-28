"""Multi-task training methods reused from the Taskonomy scaffold."""
# The Pascal-Context experiments share the exact optimization methods with the
# Taskonomy setup. We simply import the registry here so that callers can use
# the familiar entry points under the new package name.
from taskonomy_eval.methods import *  # noqa: F401,F403
from taskonomy_eval.methods import METHOD_REGISTRY  # re-export for convenience

__all__ = ["METHOD_REGISTRY"]
