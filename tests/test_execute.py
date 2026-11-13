from uuid import uuid4
import pytest

_TIMEOUT_EXIT_CODE = 124


@pytest.mark.asyncio
async def test_aexecute_command(backend):
    result = await backend.aexecute("echo 'Hello from Microsandbox'")

    assert result.exit_code == 0
    assert "Hello from Microsandbox" in result.output


@pytest.mark.asyncio
async def test_execute_command(backend):
    result = backend.execute("echo 'Hello from Microsandbox'")

    assert result.exit_code == 0
    assert "Hello from Microsandbox" in result.output


@pytest.mark.asyncio
async def test_aexecute_command_failure(backend):
    result = await backend.aexecute("exit 1")

    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_execute_command_failure(backend):
    result = backend.execute("exit 1")

    assert result.exit_code == 1


@pytest.mark.asyncio
async def test_aexecute_python(backend):
    result = await backend.aexecute("python -c 'print(1 + 1)'")

    assert result.exit_code == 0
    assert "2" in result.output


@pytest.mark.asyncio
async def test_execute_python(backend):
    result = backend.execute("python -c 'print(1 + 1)'")

    assert result.exit_code == 0
    assert "2" in result.output


@pytest.mark.asyncio
async def test_aexecute_timeout(backend):
    result = await backend.aexecute(
        "python -c 'import time; time.sleep(5);'", timeout=3
    )

    assert result.exit_code == _TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 3.0 seconds"


@pytest.mark.asyncio
async def test_execute_timeout(backend):
    result = backend.execute("python -c 'import time; time.sleep(5);'", timeout=3)

    assert result.exit_code == _TIMEOUT_EXIT_CODE
    assert result.output == "Command timed out after 3.0 seconds"


@pytest.mark.asyncio
async def test_aexecute_negative_timeout(backend):
    with pytest.raises(ValueError) as exc_info:
        result = await backend.aexecute(
            "python -c 'import time; time.sleep(5);'", timeout=-1
        )

    assert str(exc_info.value) == "timeout must be non-negative, got -1"


@pytest.mark.asyncio
async def test_execute_negative_timeout(backend):
    with pytest.raises(ValueError) as exc_info:
        result = backend.execute("python -c 'import time; time.sleep(5);'", timeout=-1)

    assert str(exc_info.value) == "timeout must be non-negative, got -1"

@pytest.mark.asyncio
async def test_aexecute_capture_offload(backend):
    capture_path = "/tmp/capture-test.txt"

    result = await backend.aexecute_with_offload(
        "python3 -c \"print('A' * 10000)\"",
        capture_path,
        max_inline_bytes=1000,
    )

    assert result.response.exit_code == 0
    assert result.offloaded is True

    captured = await backend.aread(capture_path)

    assert captured.error is None
    assert "A" * 10000 in captured.file_data["content"]

@pytest.mark.asyncio
async def test_aexecute_capture_no_offload(backend):
    capture_path = "/tmp/capture-test.txt"

    result = await backend.aexecute_with_offload(
        "python3 -c \"print('A' * 10)\"",
        capture_path,
        max_inline_bytes=1000,
    )

    assert result.response.exit_code == 0
    assert result.offloaded is False

@pytest.mark.asyncio
async def test_execute_capture_offload(backend):
    capture_path = "/tmp/capture-test.txt"

    result = backend.execute_with_offload(
        "python3 -c \"print('A' * 10000)\"",
        capture_path,
        max_inline_bytes=1000,
    )

    assert result.response.exit_code == 0
    assert result.offloaded is True

    captured =  backend.read(capture_path)

    assert captured.error is None
    assert "A" * 10000 in captured.file_data["content"]

@pytest.mark.asyncio
async def test_execute_capture_no_offload(backend):
    capture_path = "/tmp/capture-test.txt"

    result = backend.execute_with_offload(
        "python3 -c \"print('A' * 10)\"",
        capture_path,
        max_inline_bytes=1000,
    )

    assert result.response.exit_code == 0
    assert result.offloaded is False