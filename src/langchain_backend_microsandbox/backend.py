# Original work Copyright (c) 2026 LangChain
# Modifications Copyright (c) 2026 dod-ia
# SPDX-License-Identifier: MIT

"""Microsandbox backend integration for LangChain Deep Agents.

Provides a BaseSandbox implementation backed by Microsandbox.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

from deepagents.backends.protocol import (
    FILE_NOT_FOUND,
    INVALID_PATH,
    IS_DIRECTORY,
    PERMISSION_DENIED,
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox
from microsandbox import (
    ExecTimeoutError,
    FilesystemError,
    PathNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine
    from concurrent.futures import Future

    from microsandbox import ExecOutput, Sandbox

_TIMEOUT_EXIT_CODE = 124


class MicrosandboxSandbox(BaseSandbox):
    """Sandbox backend implementation powered by Microsandbox.

    Wraps an existing Microsandbox instance to expose the synchronous and
    asynchronous interfaces expected by Deep Agents (conforming to `SandboxBackendProtocol`).

    The adapter does not manage the sandbox lifecycle. The caller remains
    responsible for starting, stopping, and configuring the sandbox, as well
    as defining its persistence and cleanup policies.
    """

    def __init__(
        self,
        *,
        sandbox: Sandbox,
        sandbox_id: str,
        timeout: int,
        enable_capture_offload: bool = False,
        grep_exclude_dirs: tuple[str, ...] = (".git",),
    ) -> None:
        """Initialize the adapter with an existing Microsandbox instance.

        Prefer using the asynchronous `create()` factory, which resolves the
        sandbox identifier before constructing the adapter.

        Args:
            sandbox: Microsandbox instance to wrap.
            sandbox_id: Resolved name of the sandbox, cached for synchronous access
                through the `id` property.
            timeout: Default command execution timeout in seconds. Must be
                non-negative. A value of zero disables the execution timeout.
            enable_capture_offload: Whether to enable Deep Agents' capture-at-source
                mechanism for large command outputs. Requires a compatible shell
                environment and the utilities used by the capture wrapper.
            grep_exclude_dirs: Directory names excluded from grep searches,
                regardless of their depth in the directory tree.
        """
        self._sandbox = sandbox
        self._id = sandbox_id
        self._default_timeout = timeout
        self.enable_capture_offload = enable_capture_offload
        self.grep_exclude_dirs = grep_exclude_dirs

    @classmethod
    async def create(
        cls,
        sandbox: Sandbox,
        *,
        timeout: int = 30 * 60,
        enable_capture_offload: bool = False,
        grep_exclude_dirs: tuple[str, ...] = (".git",),
    ) -> MicrosandboxSandbox:
        """Asynchronously create an adapter around an existing Microsandbox instance.

        Resolves the sandbox name through the provider API and caches it for
        subsequent synchronous access through the `id` property.

        The sandbox image must provide Python 3 in PATH, as inherited BaseSandbox
        operations execute Python scripts inside the sandbox.

        Capture offloading is disabled by default. Enabling it requires a
        compatible shell environment and the utilities expected by Deep Agents'
        capture wrapper.

        Args:
            sandbox: Microsandbox instance to wrap.
            timeout: Default command execution timeout in seconds.
                Defaults to 1800 seconds. A value of zero disables the timeout.
            enable_capture_offload: Whether to enable capture-at-source offloading
                for large command outputs. Requires a compatible shell and
                coreutils environment.
            grep_exclude_dirs: Directory names excluded from grep searches at
                any depth. Defaults to (".git",).

        Returns:
            An initialized MicrosandboxSandbox adapter.

        Raises:
            ValueError: If `timeout` is negative.
        """
        if timeout < 0:
            msg = f"timeout must be non-negative, got {timeout}"
            raise ValueError(msg)
        # v0.6.12 exposes an awaitable property at runtime even though its stub
        # declares an async method; accept both forms across the supported range.
        name: object = sandbox.name
        if callable(name):
            name_factory = cast("Callable[[], Awaitable[str]]", name)
            sandbox_id = await name_factory()
        else:
            sandbox_id = await cast("Awaitable[str]", name)
        return cls(
            sandbox=sandbox,
            sandbox_id=sandbox_id,
            timeout=timeout,
            enable_capture_offload=enable_capture_offload,
            grep_exclude_dirs=grep_exclude_dirs,
        )

    @property
    def id(self) -> str:
        """Return the cached name of the underlying Microsandbox instance.

        The identifier is resolved asynchronously during adapter creation,
        allowing it to be exposed through the synchronous BaseSandbox interface.

        Returns:
            The sandbox name.
        """
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a shell command through the synchronous backend interface.

        Delegates execution to `aexecute()` and bridges the asynchronous provider
        API to the synchronous interface.

        Args:
            command: Shell command to execute inside the sandbox.
            timeout: Maximum execution time in seconds. If `None`, uses the
                default timeout configured when the adapter was created.
                A value of zero disables the timeout. Must be positive.

        Returns:
            An ExecuteResponse containing the combined command output and
            its exit status.
        """
        return _run_sync(lambda: self.aexecute(command, timeout=timeout))

    async def aexecute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse:
        """Execute a shell command using Microsandbox's asynchronous API.

        Uses the configured default timeout unless an execution-specific timeout
        is provided. A timeout of zero disables the provider-side timeout.

        Provider execution timeouts are converted into an ExecuteResponse with
        exit code 124, matching the behavior of other Deep Agents sandbox
        integrations.

        Args:
            command: Shell command to execute inside the sandbox.
            timeout: Maximum execution time in seconds. If `None`, uses the
                default timeout configured when the adapter was created.
                A value of zero disables the timeout. Must be positive.

        Returns:
            An ExecuteResponse containing the combined command output and its exit status.

        Raises:
            ValueError: If the effective timeout is negative.
        """
        effective_timeout = self._default_timeout if timeout is None else timeout

        if effective_timeout < 0:
            msg = f"timeout must be non-negative, got {effective_timeout}"
            raise ValueError(msg)

        effective_timeout = None if effective_timeout == 0 else float(effective_timeout)

        try:
            output = await self._sandbox.shell(command, timeout=effective_timeout)
        except ExecTimeoutError:
            msg = f"Command timed out after {effective_timeout} seconds"
            return ExecuteResponse(
                output=msg, exit_code=_TIMEOUT_EXIT_CODE, truncated=False
            )
        return _map_execute_output(output)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files through the synchronous backend interface.

        Delegates file uploads to `aupload_files()` and bridges the asynchronous
        filesystem API to the synchronous BaseSandbox interface.

        Args:
            files: List of pairs containing an absolute sandbox path and the
                corresponding file contents as bytes.

        Returns:
            A list of FileUploadResponse objects, preserving the input order.
            Individual upload failures are reported in their corresponding
            response rather than raised.
        """
        return _run_sync(lambda: self.aupload_files(files))

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files using Microsandbox's asynchronous filesystem API.

        Processes files sequentially and delegates each upload to `_aupload_file()`.

        Each file is handled independently: a failed upload does not prevent
        subsequent files from being processed.

        Args:
            files: List of pairs containing an absolute sandbox path and the
                corresponding file contents as bytes.

        Returns:
            A list of FileUploadResponse objects, preserving the input order.
            Each response contains either a successful upload result or an
            error describing the failure.
        """
        return [await self._aupload_file(path, content) for path, content in files]

    async def _aupload_file(self, path: str, content: bytes) -> FileUploadResponse:
        """Upload a single file atomically to the sandbox filesystem.

        Creates the parent directory when necessary, writes the content to a
        temporary file in the destination directory, then renames it to the
        requested path.

        Using a temporary file prevents readers from observing partially
        written file contents when the underlying filesystem supports atomic
        rename operations.

        The temporary file is removed on a best-effort basis, including when
        the upload fails.

        Args:
            path: Absolute destination path inside the sandbox.
            content: File contents to upload.

        Returns:
            A FileUploadResponse containing the destination path and either
            `None` on success or a normalized filesystem error.
        """
        if not path.startswith("/") or "\x00" in path:
            return FileUploadResponse(path=path, error=INVALID_PATH)

        temp_path = None
        try:
            parent = str(PurePosixPath(path).parent)
            if parent != "/":
                await self._sandbox.fs.mkdir(parent)
            temp_path = f"{path}.{uuid.uuid4().hex}.tmp"
            await self._sandbox.fs.write(temp_path, content)
            # `rename` is based on tokio::fs::rename (crates/agentd/lib/fs.rs)
            # This function is based on Linux rename so it is atomic in a Linux system.
            await self._sandbox.fs.rename(temp_path, path)
        except Exception as exc:  # noqa: BLE001
            return FileUploadResponse(path=path, error=_map_file_error(exc))
        finally:
            if temp_path is not None:
                with suppress(Exception):
                    await self._sandbox.fs.remove(temp_path)
        return FileUploadResponse(path=path, error=None)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files through the synchronous backend interface.

        Delegates file downloads to `adownload_files()` and bridges the
        asynchronous filesystem API to the synchronous BaseSandbox interface.

        Args:
            paths: List of absolute sandbox paths to download.

        Returns:
            A list of FileDownloadResponse objects, preserving the input order.
            Individual download failures are reported in their corresponding
            response rather than raised.
        """
        return _run_sync(lambda: self.adownload_files(paths))

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files using Microsandbox's asynchronous filesystem API.

        Processes paths sequentially and delegates each download to
        `_adownload_file()`.

        Each path is handled independently: a failed download does not prevent
        subsequent paths from being processed.

        Args:
            paths: List of absolute sandbox paths to download.

        Returns:
            A list of FileDownloadResponse objects, preserving the input order.
            Each response contains the file contents on success or a normalized
            error when the download fails.
        """
        return [await self._adownload_file(path) for path in paths]

    async def _adownload_file(self, path: str) -> FileDownloadResponse:
        """Download a single file from the sandbox filesystem.

        Reads the requested path using Microsandbox's asynchronous filesystem API.

        Args:
            path: Absolute path of the file to download inside the sandbox.

        Returns:
            A FileDownloadResponse containing the file contents on success.
            If the path is invalid or the filesystem operation fails, the
            response contains `None` as its content and a corresponding error.
        """
        if not path.startswith("/") or "\x00" in path:
            return FileDownloadResponse(path=path, content=None, error=INVALID_PATH)

        try:
            content = await self._sandbox.fs.read(path)
        except Exception as exc:  # noqa: BLE001
            return FileDownloadResponse(
                path=path, content=None, error=_map_file_error(exc)
            )
        return FileDownloadResponse(path=path, content=content, error=None)


def _map_execute_output(output: ExecOutput) -> ExecuteResponse:
    """Convert a Microsandbox execution result into a Deep Agents response.

    Decodes stdout and stderr as UTF-8, replacing invalid byte sequences
    instead of raising decoding errors.

    Stdout is preserved as the primary output. When stderr is not empty,
    it is appended using `<stderr>` tags, with a newline separator when
    necessary.

    The provider's exit code is preserved. Output truncation is not
    performed by this adapter.

    Args:
        output: Execution result returned by Microsandbox.

    Returns:
        An ExecuteResponse containing the combined output, the provider's
        exit code, and a `truncated` flag set to False.
    """
    stdout = bytes(output.stdout_bytes).decode("utf-8", errors="replace")
    stderr = bytes(output.stderr_bytes).decode("utf-8", errors="replace")

    combined = stdout
    if stderr:
        separator = "" if not stdout or stdout.endswith(("\n", "\r")) else "\n"
        combined += f"{separator}<stderr>{stderr}</stderr>"
    return ExecuteResponse(output=combined, exit_code=output.exit_code, truncated=False)


def _map_file_error(exc: Exception) -> FileOperationError | str:  # noqa: PLR0911
    """Normalize filesystem exceptions into Deep Agents file operation errors.

    Maps known Python and Microsandbox filesystem exceptions to the
    corresponding FileOperationError constants.

    For provider exceptions that do not expose structured error types,
    the exception message is inspected to identify common filesystem
    failures, such as permission errors, missing files, and invalid paths.

    Unrecognized exceptions are preserved as descriptive strings so that
    callers retain access to the original error information.

    Args:
        exc: Exception raised by a filesystem operation.

    Returns:
        A normalized FileOperationError when the failure matches a known
        category, otherwise a string containing the exception type and
        its message.
    """
    if isinstance(exc, PermissionError):
        return PERMISSION_DENIED
    if isinstance(exc, IsADirectoryError):
        return IS_DIRECTORY
    if isinstance(exc, NotADirectoryError):
        return INVALID_PATH
    if isinstance(exc, (FileNotFoundError, PathNotFoundError)):
        return FILE_NOT_FOUND

    if isinstance(exc, FilesystemError):
        message = str(exc).lower()
        substring_errors: tuple[tuple[tuple[str, ...], FileOperationError], ...] = (
            (("permission denied", "access denied", "forbidden"), PERMISSION_DENIED),
            (("is a directory",), IS_DIRECTORY),
            (("not a directory", "invalid path"), INVALID_PATH),
            (("no such file", "not found"), FILE_NOT_FOUND),
        )
        for needles, error in substring_errors:
            if any(needle in message for needle in needles):
                return error
    detail = str(exc).strip()
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _run_sync[T](factory: Callable[[], Coroutine[object, object, T]]) -> T:
    """Run an asynchronous provider operation from synchronous code.

    Bridges an asynchronous coroutine factory to the synchronous
    BaseSandbox interface.

    When no event loop is running in the calling thread, the coroutine
    is executed directly using `asyncio.run()`.

    When an event loop is already running, execution is delegated to a
    dedicated worker thread with its own event loop. This avoids attempting
    to nest `asyncio.run()` inside an existing event loop.

    The coroutine is created and executed in the same thread, ensuring
    that coroutine-bound resources are not transferred between threads.

    Args:
        factory: Zero-argument callable that creates the coroutine to execute.

    Returns:
        The result produced by the asynchronous operation.

    Raises:
        Exception: Any exception raised by the asynchronous operation is
            propagated to the synchronous caller.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    def run_in_thread() -> T:
        return asyncio.run(factory())

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="microsandbox-sync"
    ) as executor:
        future: Future[T] = executor.submit(run_in_thread)
        return future.result()
