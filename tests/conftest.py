from collections.abc import AsyncIterator
from uuid import uuid4
import pytest_asyncio
from microsandbox import Sandbox

from langchain_backend_microsandbox import MicrosandboxSandbox


@pytest_asyncio.fixture(scope="module")
async def microsandbox_vm(request) -> AsyncIterator[Sandbox]:
    """Create one Microsandbox VM per test module."""
    sandbox = await Sandbox.create(
        f"deepagents-tests-download-{request.module.__name__}-{uuid4().hex[:8]}",
        image="python:3.13",
        memory=1024,
        cpus=2,
        ephemeral=True,
    )

    try:
        yield sandbox
    finally:
        await sandbox.stop()


@pytest_asyncio.fixture(scope="module")
async def backend(
    microsandbox_vm: Sandbox,
) -> MicrosandboxSandbox:
    """Provide a Deep Agents backend using the shared VM."""
    return await MicrosandboxSandbox.create(
        microsandbox_vm,
        timeout=120,
        enable_capture_offload=True
    )
