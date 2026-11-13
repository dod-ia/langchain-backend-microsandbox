from uuid import uuid4
import pytest
from deepagents.backends.protocol import FILE_NOT_FOUND, INVALID_PATH
from microsandbox import FilesystemError, Sandbox
from langchain_backend_microsandbox import MicrosandboxSandbox

@pytest.mark.asyncio
async def test_legacy_read_write(backend, microsandbox_vm):
    content = uuid4().hex
    path = "/tmp/test-legacy.txt"

    write_result = backend.write(path, content)
    vm_content = await microsandbox_vm.fs.read(path)

    read_result = backend.read(path)

    assert write_result.error is None
    assert read_result.error is None
    assert vm_content.decode("utf-8") == content
    assert read_result.file_data["content"] == content


@pytest.mark.asyncio
async def test_legacy_aread_awrite(backend, microsandbox_vm):
    content = uuid4().hex
    path = "/tmp/test-legacy.txt"

    write_result = await backend.awrite(path, content)
    vm_content = await microsandbox_vm.fs.read(path)

    read_result = await backend.aread(path)

    assert write_result.error is None
    assert read_result.error is None
    assert vm_content.decode("utf-8") == content
    assert read_result.file_data["content"] == content


@pytest.mark.asyncio
async def test_legacy_empty_read_write(backend, microsandbox_vm):
    content = ""
    path = "/tmp/test-legacy.txt"

    write_result = backend.write(path, content)
    vm_content = await microsandbox_vm.fs.read(path)

    read_result = backend.read(path)

    assert write_result.error is None
    assert read_result.error is None
    assert vm_content.decode("utf-8") == ""
    assert "File exists but has empty contents" in read_result.file_data["content"]


@pytest.mark.asyncio
async def test_legacy_empty_aread_awrite(backend, microsandbox_vm):
    content = ""
    path = "/tmp/test-legacy.txt"

    write_result = await backend.awrite(path, content)
    vm_content = await microsandbox_vm.fs.read(path)

    read_result = await backend.aread(path)

    assert write_result.error is None
    assert read_result.error is None
    assert vm_content.decode("utf-8") == ""
    assert "File exists but has empty contents" in read_result.file_data["content"]


@pytest.mark.asyncio
async def test_legacy_write_overwrite(backend, microsandbox_vm):
    content_old = uuid4().hex
    content_new = uuid4().hex
    path = "/tmp/test-legacy.txt"

    backend.write(path, content_old)
    write_result = backend.write(path, content_new)

    vm_content = await microsandbox_vm.fs.read(path)

    assert write_result.error is None
    assert vm_content.decode("utf-8") == content_new


@pytest.mark.asyncio
async def test_legacy_read_file_not_found(backend):
    path = "/tmp/nonexistent-file.txt"

    result = backend.read(path)

    assert result.error is not None
    assert result.error == "File '/tmp/nonexistent-file.txt': file_not_found"


@pytest.mark.asyncio
async def test_legacy_aread_file_not_found(backend):
    path = "/tmp/nonexistent-file.txt"

    result = await backend.aread(path)

    assert result.error is not None
    assert result.error == "File '/tmp/nonexistent-file.txt': file_not_found"


@pytest.mark.asyncio
async def test_legacy_aedit(backend, microsandbox_vm):
    path = "/tmp/test-edit.txt"
    content = b"Hello World\nThis is a test"

    await microsandbox_vm.fs.write(path, content)

    result = await backend.aedit(
        path,
        old_string="Hello World",
        new_string="Hello Microsandbox",
        replace_all=False,
    )

    assert result.error is None

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == b"Hello Microsandbox\nThis is a test"


@pytest.mark.asyncio
async def test_legacy_edit(backend, microsandbox_vm):
    path = "/tmp/test-edit.txt"
    content = b"Hello World\nThis is a test"

    await microsandbox_vm.fs.write(path, content)

    result = backend.edit(
        path,
        old_string="Hello World",
        new_string="Hello Microsandbox",
        replace_all=False,
    )

    assert result.error is None

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == b"Hello Microsandbox\nThis is a test"


@pytest.mark.asyncio
async def test_legacy_aedit_string_not_found(backend, microsandbox_vm):
    path = "/tmp/test-edit-not-found.txt"

    await microsandbox_vm.fs.write(path, b"Hello World")

    result = await backend.aedit(
        path,
        old_string="Goodbye",
        new_string="Hello",
    )

    assert result.error is not None
    assert "String not found in file" in result.error

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == b"Hello World"


@pytest.mark.asyncio
async def test_legacy_edit_string_not_found(backend, microsandbox_vm):
    path = "/tmp/test-edit-not-found.txt"

    await microsandbox_vm.fs.write(path, b"Hello World")

    result = backend.edit(
        path,
        old_string="Goodbye",
        new_string="Hello",
    )

    assert result.error is not None
    assert "String not found in file" in result.error

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == b"Hello World"


@pytest.mark.asyncio
async def test_legacy_aedit_replace_all(backend, microsandbox_vm):
    path = "/tmp/test-edit-all.txt"

    await microsandbox_vm.fs.write(
        path,
        b"Hello World\nHello World\nHello World",
    )

    result = await backend.aedit(
        path,
        old_string="Hello World",
        new_string="Hello Microsandbox",
        replace_all=True,
    )

    assert result.error is None

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == (
        b"Hello Microsandbox\nHello Microsandbox\nHello Microsandbox"
    )


@pytest.mark.asyncio
async def test_legacy_edit_replace_all(backend, microsandbox_vm):
    path = "/tmp/test-edit-all.txt"

    await microsandbox_vm.fs.write(
        path,
        b"Hello World\nHello World\nHello World",
    )

    result = backend.edit(
        path,
        old_string="Hello World",
        new_string="Hello Microsandbox",
        replace_all=True,
    )

    assert result.error is None

    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == (
        b"Hello Microsandbox\nHello Microsandbox\nHello Microsandbox"
    )


@pytest.mark.asyncio
async def test_backend_edit_multiple_matches(backend, microsandbox_vm):
    path = "/tmp/test-edit-multiple.txt"

    await microsandbox_vm.fs.write(
        path,
        b"Hello World\nHello World\nHello World",
    )

    result = backend.edit(
        path,
        old_string="Hello World",
        new_string="Hello Microsandbox",
        replace_all=False,
    )

    assert result.error is not None
    assert "Use replace_all=True to replace all occurrences" in result.error
    actual_content = await microsandbox_vm.fs.read(path)

    assert actual_content == b"Hello World\nHello World\nHello World"


@pytest.mark.asyncio
async def test_legacy_delete(backend, microsandbox_vm):
    path = "/tmp/test-delete.txt"

    await microsandbox_vm.fs.write(path, uuid4().hex.encode("utf-8"))

    result = backend.delete(path)

    assert result.error is None

    with pytest.raises(FilesystemError) as exc:
        await microsandbox_vm.fs.read(path)
    assert "No such file or directory" in str(exc.value)


def test_legacy_delete_file_not_found(backend):
    path = "/tmp/nonexistent-delete.txt"

    result = backend.delete(path)

    assert result.error is not None
    assert "not found" in str(result.error)


@pytest.mark.asyncio
async def test_legacy_agrep(backend, microsandbox_vm):
    directory = "/tmp/test-grep"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Hello Microsandbox\nGoodbye World",
    )

    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Hello Deep Agents\nAnother line",
    )

    result = await backend.agrep(
        pattern="Hello",
        path=directory,
    )

    assert result.error is None
    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_legacy_grep(backend, microsandbox_vm):
    directory = "/tmp/test-grep"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Hello Microsandbox\nGoodbye World",
    )

    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Hello Deep Agents\nAnother line",
    )

    result = backend.grep(
        pattern="Hello",
        path=directory,
    )

    assert result.error is None
    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_legacy_agrep_no_match(backend, microsandbox_vm):
    path = "/tmp/test-grep-empty.txt"

    await microsandbox_vm.fs.write(
        path,
        b"Hello World",
    )

    result = await backend.agrep(
        pattern="Goodbye",
        path=path,
    )

    assert result.error is None
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_legacy_grep_no_match(backend, microsandbox_vm):
    path = "/tmp/test-grep-empty.txt"

    await microsandbox_vm.fs.write(
        path,
        b"Hello World",
    )

    result = backend.grep(
        pattern="Goodbye",
        path=path,
    )

    assert result.error is None
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_legacy_agrep_file_not_found(backend):
    result = await backend.agrep(
        pattern="Hello",
        path="tmp/nonexistent-grep.txt",
    )
    assert result.error is None
    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_legacy_grep_file_not_found(backend):
    result = backend.grep(
        pattern="Hello",
        path="tmp/nonexistent-grep.txt",
    )
    assert result.error is None
    assert len(result.matches) == 0

@pytest.mark.asyncio
async def test_agrep_excludes_git_by_default(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(microsandbox_vm)

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = await backend.agrep("unique_marker", path="/workspace")

    assert result.matches is not None
  
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert "/workspace/.git/config" not in paths

@pytest.mark.asyncio
async def test_grep_excludes_git_by_default(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(microsandbox_vm)

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = backend.grep("unique_marker", path="/workspace")

    assert result.matches is not None
  
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert "/workspace/.git/config" not in paths

@pytest.mark.asyncio
async def test_agrep_excludes_custom_directories(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(".git", "node_modules"),
    )

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite(
        "/workspace/node_modules/package/index.js",
        "unique_marker",
    )

    result = await backend.agrep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert not any("/node_modules/" in path for path in paths)

@pytest.mark.asyncio
async def test_grep_excludes_custom_directories(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(".git", "node_modules"),
    )

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite(
        "/workspace/node_modules/package/index.js",
        "unique_marker",
    )

    result = backend.grep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert not any("/node_modules/" in path for path in paths)

@pytest.mark.asyncio
async def test_agrep_excludes_directories_at_any_depth(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=("generated",),
    )

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite(
        "/workspace/src/generated/output.py",
        "unique_marker",
    )
    await backend.awrite(
        "/workspace/src/others/output.py",
        "unique_marker",
    )

    result = await backend.agrep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert "/workspace/src/others/output.py" in paths
    assert not any("/generated/" in path for path in paths)

@pytest.mark.asyncio
async def test_grep_excludes_directories_at_any_depth(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=("generated",),
    )

    await backend.awrite("/workspace/app.py", "unique_marker")
    await backend.awrite(
        "/workspace/src/generated/output.py",
        "unique_marker",
    )
    await backend.awrite(
        "/workspace/src/others/output.py",
        "unique_marker",
    )

    result = backend.grep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/app.py" in paths
    assert "/workspace/src/others/output.py" in paths
    assert not any("/generated/" in path for path in paths)

@pytest.mark.asyncio
async def test_agrep_empty_exclusions_searches_everything(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(),
    )

    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = await backend.agrep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/.git/config" in paths

@pytest.mark.asyncio
async def test_grep_empty_exclusions_searches_everything(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(),
    )

    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = backend.grep("unique_marker", path="/workspace")

    assert result.matches is not None
    paths = [match["path"] for match in result.matches]

    assert "/workspace/.git/config" in paths

@pytest.mark.asyncio
async def test_agrep_can_search_explicitly_excluded_directory(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(".git",),
    )

    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = await backend.agrep(
        "unique_marker",
        path="/workspace/.git",
    )

    assert result.matches is not None
    assert any(
        match["path"].endswith("/.git/config")
        for match in result.matches
    )

@pytest.mark.asyncio
async def test_grep_can_search_explicitly_excluded_directory(
    microsandbox_vm: Sandbox,
) -> None:
    backend = await MicrosandboxSandbox.create(
        microsandbox_vm,
        grep_exclude_dirs=(".git",),
    )

    await backend.awrite("/workspace/.git/config", "unique_marker")

    result = backend.grep(
        "unique_marker",
        path="/workspace/.git",
    )

    assert result.matches is not None
    assert any(
        match["path"].endswith("/.git/config")
        for match in result.matches
    )

@pytest.mark.asyncio
async def test_legacy_aglob(backend, microsandbox_vm):
    directory = "/tmp/test-glob"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Content 1",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Content 2",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file3.py",
        b"Content 3",
    )

    result = await backend.aglob(
        pattern="*.txt",
        path=directory,
    )

    assert result.error is None
    assert len(result.matches) == 2

    paths = [entry["path"] for entry in result.matches]

    assert f"{directory}/file1.txt" in paths
    assert f"{directory}/file2.txt" in paths

    assert all(entry["is_dir"] is False for entry in result.matches)


@pytest.mark.asyncio
async def test_legacy_glob(backend, microsandbox_vm):
    directory = "/tmp/test-glob"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Content 1",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Content 2",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file3.py",
        b"Content 3",
    )

    result = backend.glob(
        pattern="*.txt",
        path=directory,
    )

    assert result.error is None
    assert len(result.matches) == 2

    paths = [entry["path"] for entry in result.matches]

    assert f"{directory}/file1.txt" in paths
    assert f"{directory}/file2.txt" in paths


@pytest.mark.asyncio
async def test_legacy_aglob_no_match(backend, microsandbox_vm):
    directory = "/tmp/test-glob-empty"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file.txt",
        b"Content",
    )

    result = await backend.aglob(
        pattern="*.py",
        path=directory,
    )

    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_legacy_glob_no_match(backend, microsandbox_vm):
    directory = "/tmp/test-glob-empty"

    await microsandbox_vm.fs.mkdir(directory)

    await microsandbox_vm.fs.write(
        f"{directory}/file.txt",
        b"Content",
    )

    result = backend.glob(
        pattern="*.py",
        path=directory,
    )

    assert len(result.matches) == 0


@pytest.mark.asyncio
async def test_legacy_als(backend, microsandbox_vm):
    directory = "/tmp/test-ls"

    await microsandbox_vm.fs.mkdir(directory)
    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Content 1",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Content 2",
    )
    await microsandbox_vm.fs.mkdir(f"{directory}/subdir")

    result = await backend.als(directory)

    assert result.error is None
    assert len(result.entries) == 3

    entries = {entry["path"]: entry for entry in result.entries}

    assert entries[f"{directory}/file1.txt"]["is_dir"] is False
    assert entries[f"{directory}/file2.txt"]["is_dir"] is False
    assert entries[f"{directory}/subdir"]["is_dir"] is True


@pytest.mark.asyncio
async def test_legacy_ls(backend, microsandbox_vm):
    directory = "/tmp/test-ls"

    await microsandbox_vm.fs.mkdir(directory)
    await microsandbox_vm.fs.write(
        f"{directory}/file1.txt",
        b"Content 1",
    )
    await microsandbox_vm.fs.write(
        f"{directory}/file2.txt",
        b"Content 2",
    )
    await microsandbox_vm.fs.mkdir(f"{directory}/subdir")

    result = backend.ls(directory)

    assert result.error is None
    assert len(result.entries) == 3

    entries = {entry["path"]: entry for entry in result.entries}

    assert entries[f"{directory}/file1.txt"]["is_dir"] is False
    assert entries[f"{directory}/file2.txt"]["is_dir"] is False
    assert entries[f"{directory}/subdir"]["is_dir"] is True
