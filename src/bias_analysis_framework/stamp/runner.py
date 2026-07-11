"""Subprocess helpers for invoking the external STAMP CLI."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StampCommandResult:
    """Result of one STAMP command."""

    command: tuple[str, ...]
    config_path: Path
    log_path: Path
    return_code: int
    executed: bool

    @property
    def succeeded(self) -> bool:
        """Return whether the command completed successfully."""
        return self.executed and self.return_code == 0


def normalize_stamp_command(command: str | Sequence[str]) -> list[str]:
    """Convert a configured command into a subprocess argument list.

    Examples
    --------
    ``"stamp"`` becomes ``["stamp"]``.

    ``"uv run stamp"`` becomes ``["uv", "run", "stamp"]``.
    """
    if isinstance(command, str):
        parts = shlex.split(command)
    else:
        parts = [str(part) for part in command]

    if not parts:
        raise ValueError("STAMP command cannot be empty.")

    return parts


def build_stamp_preprocess_command(
    command: str | Sequence[str],
    config_path: str | Path,
) -> list[str]:
    """Build the STAMP preprocessing CLI command."""
    config_path = Path(config_path).expanduser().resolve()

    return [
        *normalize_stamp_command(command),
        "--config",
        str(config_path),
        "preprocess",
    ]


def check_stamp_command_available(
    command: str | Sequence[str],
) -> None:
    """Check whether the configured STAMP executable is available."""
    command_parts = normalize_stamp_command(command)
    executable = command_parts[0]

    if Path(executable).expanduser().is_file():
        return

    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"STAMP command executable was not found: {executable!r}. "
            "Activate the STAMP environment or configure stamp.command "
            "with the correct executable."
        )


def build_stamp_environment(
    *,
    token_file: str | Path | None = None,
    cache_environment: Mapping[str, str | Path] | None = None,
    extra_environment: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the environment used for a STAMP subprocess.

    The token is read at runtime and is never written to a generated YAML
    config or log file.
    """
    environment = os.environ.copy()

    if token_file is not None:
        token_path = Path(token_file).expanduser().resolve()

        if not token_path.exists():
            raise FileNotFoundError(
                f"Hugging Face token file does not exist: {token_path}"
            )

        token = token_path.read_text(encoding="utf-8").strip()

        if not token:
            raise ValueError(f"Hugging Face token file is empty: {token_path}")

        environment["HF_TOKEN"] = token
        environment["HUGGINGFACE_HUB_TOKEN"] = token

    if cache_environment:
        for variable, value in cache_environment.items():
            path = Path(value).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            environment[str(variable)] = str(path)

    if extra_environment:
        for variable, value in extra_environment.items():
            environment[str(variable)] = str(value)

    return environment


def run_stamp_preprocess_command(
    *,
    command: str | Sequence[str],
    config_path: str | Path,
    log_path: str | Path,
    environment: Mapping[str, str] | None = None,
    working_directory: str | Path | None = None,
    execute: bool = False,
) -> StampCommandResult:
    """Run or preview one STAMP preprocessing command.

    When ``execute`` is false, the command is written to the log but is not
    executed. This is the recommended first test.
    """
    config_path = Path(config_path).expanduser().resolve()
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command_args = build_stamp_preprocess_command(command, config_path)
    printable_command = shlex.join(command_args)

    if not execute:
        log_path.write_text(
            "DRY RUN\n"
            f"Command: {printable_command}\n"
            f"Config: {config_path}\n",
            encoding="utf-8",
        )

        return StampCommandResult(
            command=tuple(command_args),
            config_path=config_path,
            log_path=log_path,
            return_code=0,
            executed=False,
        )

    check_stamp_command_available(command)

    cwd = (
        Path(working_directory).expanduser().resolve()
        if working_directory is not None
        else None
    )

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {printable_command}\n")
        log_file.write(f"Config: {config_path}\n\n")
        log_file.flush()

        completed = subprocess.run(
            command_args,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(environment) if environment is not None else None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=completed.returncode,
            cmd=command_args,
        )

    return StampCommandResult(
        command=tuple(command_args),
        config_path=config_path,
        log_path=log_path,
        return_code=completed.returncode,
        executed=True,
    )