from __future__ import annotations

import nox
from nox.command import CommandFailed

PYTHON = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# (pin, allow_pre)
POLARS = [  # latest *
    ("polars==1.32.*", False),
    ("polars==1.33.*", False),
    ("polars==1.34.*", False),
    ("polars==1.35.*", False),
    # (
    #     "polars>=1.34b4,<1.35",
    #     True,
    # ),  # any 1.34 pre (a/b/rc); skip if none published
]

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True


@nox.session(python=PYTHON)
@nox.parametrize(("polars_pin", "allow_pre"), POLARS)
def tests(session, polars_pin, allow_pre):
    # Build installer args
    install_cmd = ["uv", "pip", "install"]
    if allow_pre:
        install_cmd.append("--pre")  # allow alpha/beta/rc
    install_cmd += ["-e", ".", polars_pin, "pytest", "pytest-xdist"]

    try:
        session.run(*install_cmd, external=True)
    except CommandFailed:
        if allow_pre:
            session.skip(
                "No 1.34 prerelease available (skipping this combo)."
            )
        raise

    session.run("pytest", "-q")  # or: "-n", "auto" if using xdist
