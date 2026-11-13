from uuid import uuid4
import pytest
from deepagents.backends.protocol import FILE_NOT_FOUND, INVALID_PATH


@pytest.mark.asyncio
async def test_adownload_file(backend, microsandbox_vm):
    content = uuid4().hex.encode("utf-8")
    path = "/tmp/test-adownload.txt"

    # Création directe du fichier dans Microsandbox
    await microsandbox_vm.fs.write(path, content)

    # Téléchargement via Deep Agents
    download = await backend.adownload_files([path])

    assert download[0].error is None
    assert download[0].content == content


@pytest.mark.asyncio
async def test_download_file(backend, microsandbox_vm):
    content = uuid4().hex.encode("utf-8")
    path = "/tmp/test-download.txt"

    # Création directe du fichier dans Microsandbox
    await microsandbox_vm.fs.write(path, content)

    # Téléchargement via Deep Agents
    download = backend.download_files([path])

    assert download[0].error is None
    assert download[0].content == content


@pytest.mark.asyncio
async def test_adownload_file_not_found(backend):
    result = await backend.adownload_files(["/tmp/nonexistent-file.txt"])

    assert result[0].error is not None
    assert result[0].error == FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_download_file_not_found(backend):
    result = backend.download_files(["/tmp/nonexistent-file.txt"])

    assert result[0].error is not None
    assert result[0].error == FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_adownload_invalid_file(backend):
    result = await backend.adownload_files(["tmp/nonexistent-file.txt"])

    assert result[0].error == INVALID_PATH


@pytest.mark.asyncio
async def test_download_invalid_file(backend):
    result = backend.download_files(["tmp/nonexistent-file.txt"])

    assert result[0].error == INVALID_PATH
