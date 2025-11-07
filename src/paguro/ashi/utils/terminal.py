from __future__ import annotations

import os
import sys
from functools import lru_cache


class TerminalDetector:
    __slots__ = ("_env",)

    def __init__(self):
        self._env = self._get_environment_vars()

    def __str__(self) -> str:
        return f"<TerminalDetector: term_program={self.term_program}, color_system={self.color_system}, supports_styling={self.supports_styling}>"

    def _get_environment_vars(self) -> dict[str, str]:
        return {
            "TERM": os.getenv("TERM", "").lower(),
            "COLORTERM": os.getenv("COLORTERM", "").lower(),
            "NO_COLOR": os.getenv("NO_COLOR", ""),
            "FORCE_COLOR": os.getenv("FORCE_COLOR", ""),
            "TERM_PROGRAM": os.getenv("TERM_PROGRAM", ""),
            "TERM_PROGRAM_VERSION": os.getenv(
                "TERM_PROGRAM_VERSION", ""
            ).lower(),
            "JUPYTER_EXECUTING": os.getenv("JUPYTER_EXECUTING", ""),
            "JPY_PARENT_PID": os.getenv("JPY_PARENT_PID", ""),
            "IPYTHON": os.getenv("IPYTHON", ""),
            "PYCHARM_HOSTED": os.getenv("PYCHARM_HOSTED", ""),
            "PYTHONIOENCODING": os.getenv("PYTHONIOENCODING", ""),
            "CI": os.getenv("CI", ""),
            "GITHUB_ACTIONS": os.getenv("GITHUB_ACTIONS", ""),
            "GITLAB_CI": os.getenv("GITLAB_CI", ""),
            "JENKINS_URL": os.getenv("JENKINS_URL", ""),
            "TRAVIS": os.getenv("TRAVIS", ""),
            "SSH_CLIENT": os.getenv("SSH_CLIENT", ""),
            "SSH_TTY": os.getenv("SSH_TTY", ""),
            "TMUX": os.getenv("TMUX", ""),
            "STY": os.getenv("STY", ""),
        }

    @property
    @lru_cache
    def is_jupyter(self) -> bool:
        return bool(
            self._env["JUPYTER_EXECUTING"]
            or self._env["JPY_PARENT_PID"]
            or (self._env["IPYTHON"] and "IPKernelApp" in sys.modules)
            or "jupyter_client" in sys.modules
            or "ipykernel" in sys.modules
        )

    @property
    @lru_cache
    def is_unix(self) -> bool:
        """Check if the system is a UNIX-based system (Linux, macOS, etc.)."""
        return sys.platform not in ("win32", "cygwin")

    @property
    @lru_cache
    def is_windows(self) -> bool:
        """Check if the platform is Windows (including WSL)."""
        return sys.platform == "win32"

    @property
    @lru_cache
    def is_wsl(self) -> bool:
        return (
            "microsoft" in self._env["TERM_PROGRAM_VERSION"]
            or "wsl" in os.uname().release.lower()
        )

    @property
    @lru_cache
    def is_ide_console(self) -> bool:
        return bool(
            self._env["PYCHARM_HOSTED"]
            or "SPYDER_KERNEL_FILE" in os.environ
            or "VSCODE_PID" in os.environ
        )

    @property
    @lru_cache
    def is_ci_cd(self) -> bool:
        return any(
            self._env[key]
            for key in [
                "CI",
                "GITHUB_ACTIONS",
                "GITLAB_CI",
                "JENKINS_URL",
                "TRAVIS",
            ]
        )

    @property
    @lru_cache
    def is_docker(self) -> bool:
        return os.path.exists("/.dockerenv")

    @property
    @lru_cache
    def is_ssh(self) -> bool:
        return bool(self._env["SSH_CLIENT"] or self._env["SSH_TTY"])

    @property
    @lru_cache
    def is_terminal_multiplexer(self) -> bool:
        return bool(self._env["TMUX"] or self._env["STY"])

    @property
    @lru_cache
    def is_terminal(self) -> bool:
        return (
            self.is_jupyter
            or self.is_ide_console
            or self.is_ci_cd
            or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
        )

    @property
    @lru_cache
    def supports_styling(self) -> bool:
        if self._env["NO_COLOR"]:
            return False
        if (
            self._env["FORCE_COLOR"]
            or self.is_ide_console
            or self.is_jupyter
        ):
            return True
        if not self.is_terminal:
            return False
        term = self._env["TERM"]
        return (
            term
            in (
                "xterm",
                "vt100",
                "vt220",
                "xterm-color",
                "xterm-256color",
                "screen",
                "screen-256color",
                "linux",
                "cygwin",
            )
            or "256color" in term
            or self._env["COLORTERM"] in ("truecolor", "24bit")
            or "iTerm" in self._env["TERM_PROGRAM"]
            or self._env["PYTHONIOENCODING"]
            .lower()
            .startswith(("utf-8", "utf8"))
        )

    @property
    @lru_cache
    def color_system(self) -> str | None:
        if not self.supports_styling:
            return None
        if self.is_jupyter or self.is_ide_console:
            return "truecolor"
        if sys.platform == "win32":
            return self._detect_windows_color_system()
        return self._detect_unix_color_system()

    def _detect_windows_color_system(self) -> str:
        if sys.getwindowsversion().build >= 14393:  # type: ignore[attr-defined]
            return "truecolor"
        if "ANSICON" in os.environ:
            return "ansi"
        if "ConEmuANSI" in os.environ or "WT_SESSION" in os.environ:
            return "truecolor"
        return "standard"

    def _detect_unix_color_system(self) -> str:
        if self._env["COLORTERM"] in ("truecolor", "24bit"):
            return "truecolor"
        if "256" in self._env["TERM"] or self._env["TERM"] in (
            "xterm",
            "screen",
        ):
            return "256"
        return "standard"

    @property
    @lru_cache
    def term_program(self) -> str | None:
        if self.is_ide_console:
            return "IDE Console"
        if self.is_ci_cd:
            return "CI/CD Environment"
        if self.is_docker:
            return "Docker Container"
        return self._env["TERM_PROGRAM"]

    @property
    @lru_cache
    def term_version(self) -> str | None:
        return self._env["TERM_PROGRAM_VERSION"]

    @property
    @lru_cache
    def supports_powerline_fonts(self) -> bool:
        if not self.is_terminal:
            return False
        try:
            import locale

            encoding = locale.getpreferredencoding()
            "\u2b80".encode(encoding)
            return True
        except UnicodeEncodeError:
            return False

    def get_terminal_size(self) -> tuple | None:
        try:
            return os.get_terminal_size()
        except (AttributeError, OSError):
            return None
            # return (80, 24)

    def get_terminal_width(self) -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80
            # return (80, 24)

    @lru_cache
    def supports_unicode(self) -> bool:
        encoding = sys.stdout.encoding or "utf-8"
        return encoding.lower().startswith(("utf-8", "utf8"))

    @lru_cache
    def get_color_depth(self) -> int:
        if self.color_system == "truecolor":
            return 16_777_216
        elif self.color_system == "256":
            return 256
        elif self.color_system == "standard":
            return 16
        return 1

    @lru_cache
    def get_capabilities(self) -> list[str]:
        capabilities = []
        for attr in dir(self):
            if attr.startswith("is_") or attr.startswith("supports_"):
                if getattr(self, attr):
                    capabilities.append(attr)
        if self.color_system:
            capabilities.append(f"color_system_{self.color_system}")
        return capabilities

    @property
    @lru_cache
    def ansi_support_level(self) -> str:
        if not self.supports_styling:
            return "none"
        mapping = {"truecolor": "full", "256": "high", "standard": "basic"}
        key = self.color_system or ""  # ensure a str
        return mapping.get(key, "none")

        # return {"truecolor": "full", "256": "high", "standard": "basic"}.get(
        #     self.color_system, "none"
        # )

    def print_info(self):
        print(f"Is terminal: {self.is_terminal}")
        print(f"Is Unix: {self.is_unix}")
        print(f"Is Jupyter: {self.is_jupyter}")
        print(f"Is IDE console: {self.is_ide_console}")
        print(f"Is CI/CD environment: {self.is_ci_cd}")
        print(f"Is Docker container: {self.is_docker}")
        print(f"Is SSH session: {self.is_ssh}")
        print(f"Is terminal multiplexer: {self.is_terminal_multiplexer}")
        print(f"Supports styling: {self.supports_styling}")
        print(f"Color system: {self.color_system}")
        print(f"Terminal program: {self.term_program}")
        print(f"Terminal version: {self.term_version}")
        print(f"Terminal size: {self.get_terminal_size()}")
        print(f"Supports Unicode: {self.supports_unicode()}")
        print(f"Supports Powerline fonts: {self.supports_powerline_fonts}")
        print(f"Color depth: {self.get_color_depth()}")
        print(f"ANSI support level: {self.ansi_support_level}")
        print(f"Capabilities: {', '.join(self.get_capabilities())}")


# def terminal_supports_styling():
#     """
#     Detects if the terminal supports ANSI styling.
#     Returns:
#         bool: True if the terminal supports ANSI styling, False otherwise.
#     """
#     if not sys.stdout.isatty():
#         return False
#
#     term = os.getenv("TERM", "").lower()
#     colorterm = os.getenv("COLORTERM", "").lower()
#     no_color = os.getenv("NO_COLOR", "")
#     force_color = os.getenv("FORCE_COLOR", "")
#
#     if no_color:
#         return False
#
#     if force_color:
#         return True
#
#     if term in (
#             "xterm", "vt100", "vt220", "xterm-color", "xterm-256color", "screen",
#             "screen-256color",
#             "linux", "cygwin"):
#         return True
#
#     if "256color" in term or colorterm in ("truecolor", "24bit"):
#         return True
#
#     if "iTerm.app" in os.getenv("TERM_PROGRAM", "") or "iterm2" in os.getenv(
#             "TERM_PROGRAM_VERSION", "").lower():
#         return True
#
#     # Check for IPython or Jupyter environments
#     try:
#         if 'IPython' in sys.modules:
#             print("IPYTHON")
#             from IPython import get_ipython
#             ipython = get_ipython()
#             if ipython:
#                 if 'IPKernelApp' in ipython.config:  # Jupyter notebook or qtconsole
#                     return True
#                 if 'TerminalInteractiveShell' in ipython.config:  # Terminal-based IPython
#                     return True
#     except ImportError:
#         pass
#
#     return False
