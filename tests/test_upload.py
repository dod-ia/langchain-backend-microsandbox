from uuid import uuid4
import pytest

from deepagents.backends.protocol import INVALID_PATH


@pytest.mark.asyncio
async def test_aupload_file(backend, microsandbox_vm):
    content = uuid4().hex.encode("utf-8")
    path = "/tmp/test-aupload.txt"

    upload = await backend.aupload_files([(path, content)])

    assert upload[0].error is None
    vm_content = await microsandbox_vm.fs.read(path)

    assert vm_content == content


@pytest.mark.asyncio
async def test_upload_file(backend, microsandbox_vm):
    content = uuid4().hex.encode("utf-8")
    path = "/tmp/test-upload.txt"

    upload = backend.upload_files([(path, content)])

    assert upload[0].error is None
    vm_content = await microsandbox_vm.fs.read(path)

    assert vm_content == content


@pytest.mark.asyncio
async def test_empty_aupload_file(backend, microsandbox_vm):
    content = b""
    path = "/tmp/test-aupload.txt"

    upload = await backend.aupload_files([(path, content)])

    assert upload[0].error is None
    vm_content = await microsandbox_vm.fs.read(path)

    assert vm_content == content


@pytest.mark.asyncio
async def test_empty_upload_file(backend, microsandbox_vm):
    content = b""
    path = "/tmp/test-upload.txt"

    upload = backend.upload_files([(path, content)])

    assert upload[0].error is None
    vm_content = await microsandbox_vm.fs.read(path)

    assert vm_content == content


@pytest.mark.asyncio
async def test_invalid_aupload_file(backend, microsandbox_vm):
    content = b""
    path = "tmp/test-upload.txt"

    upload = await backend.aupload_files([(path, content)])

    assert upload[0].error == INVALID_PATH


@pytest.mark.asyncio
async def test_invalid_upload_file(backend, microsandbox_vm):
    content = b""
    path = "tmp/test-upload.txt"

    upload = backend.upload_files([(path, content)])

    assert upload[0].error == INVALID_PATH
