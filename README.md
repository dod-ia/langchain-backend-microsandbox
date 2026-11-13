# langchain-backend-microsandbox

**A [Microsandbox](https://github.com/superradcompany/microsandbox) backend for [LangChain Deep Agents](https://github.com/langchain-ai/deepagents), bringing isolated microVM execution to autonomous AI agents.**

Run agent-generated commands and manage files inside lightweight, isolated virtual machines through Deep Agents' native BaseSandbox interface.

Built on Microsandbox, this integration provides a seamless bridge between Deep Agents and Microsandbox's asynchronous execution and filesystem APIs.

Inspired by `kenwoodjw/langchain-microsandbox`, this implementation extends the original approach with a focus on compatibility, predictable behavior, and closer adherence to the BaseSandbox contract.

## Why Microsandbox?

AI agents increasingly need to execute code, manipulate files, and interact with shell environments to accomplish complex tasks.

Running these operations directly on the host introduces unnecessary security risks.

Microsandbox addresses this challenge by providing isolated microVM environments in which agent workloads can execute independently of the host system.

langchain-backend-microsandbox brings this isolation model to LangChain Deep Agents, allowing developers to use Microsandbox as their agent execution backend without implementing their own sandbox integration.

**The result: a standardized Deep Agents backend with isolated execution and native filesystem support.**

## Features

| Feature | Description |
|---|---|
| **Deep Agents integration** | Implements the `BaseSandbox` interface. |
| **Contract compliance** | Designed to closely follow the `BaseSandbox` behavior and expectations. |
| **Isolated execution** | Runs agent-generated commands inside Microsandbox microVMs. |
| **Async & sync execution** | Supports both synchronous and asynchronous command execution. |
| **Native file transfers** | Binary-safe file uploads and downloads through Microsandbox's filesystem API. |
| **Filesystem operations** | Inherits `read`, `write`, `edit`, `delete`, `ls`, `grep`, and `glob` from `BaseSandbox`. |
| **Timeout management** | Configurable execution timeouts with standardized timeout exit codes. |
| **Error normalization** | Maps filesystem exceptions to Deep Agents-compatible error codes. |
| **Atomic file uploads** | Uses temporary files and filesystem rename operations to avoid exposing partially written files. |

## Architecture & Compatibility

MicrosandboxSandbox acts as an adapter between LangChain Deep Agents and an existing Microsandbox instance.

It implements the backend-specific operations while relying on BaseSandbox for the higher-level filesystem functionality.

### Inherited BaseSandbox operations

The following operations are inherited directly from LangChain Deep Agents:

* `read()`, `write()`, `edit()`, `delete()`
* `ls()`, `grep()`, `glob()`

This approach avoids reimplementing filesystem behavior and helps maintain consistency with other Deep Agents sandbox backends.

### BaseSandbox behavior

todo

## Inherited BaseSandbox Behavior

MicrosandboxSandbox inherits the default filesystem operations from LangChain Deep Agents' BaseSandbox, preserving the following behaviors:

* Absolute paths: read() requires absolute paths.
  * `file_path - str - Absolute path to the file to read.`
  * Source: https://reference.langchain.com/python/deepagents/backends/sandbox/BaseSandbox/read
* Grep exclusions: Directories configured in grep_exclude_dirs are excluded from recursive searches at any depth. However, explicitly targeting an excluded directory still allows searching its contents.
  * `A search rooted at one of these directories still searches it.`
  * Source: https://reference.langchain.com/python/deepagents/backends/sandbox/BaseSandbox/grep_exclude_dirs
* **Timeout handling**: A timeout of 0 disables command execution timeouts. Negative timeout values are rejected.
  * `A value of 0 may disable timeouts on backends that support no-timeout execution.`
  * Source: https://reference.langchain.com/python/deepagents/backends/protocol/SandboxBackendProtocol/execute
* **Python in the VM**: The environment must have a Python interpreter
  
## Requirements

- Python >= 3.12
- [Deep Agents](https://github.com/langchain-ai/deepagents) >= 0.7.35, < 0.8.0
- [Microsandbox](https://github.com/superradcompany/microsandbox) >= 0.7.14, < 0.8.0
- A supported Microsandbox runtime environment.

Microsandbox requires hardware virtualization support:
- **Linux**: KVM
- **macOS**: Apple Silicon virtualization support
- **Windows**: Windows Hypervisor Platform (WHP)

On **debian/ubuntu system**, you can install KVM  by installing:

`sudo apt install qemu-system libvirt-daemon-system`

## Installation from sources

Clone the repository and install the package:

```bash
git clone https://github.com/dod-ia/langchain-backend-microsandbox.git

cd langchain-backend-microsandbox

uv sync
```

## Get started

The adapter wraps an existing Microsandbox instance and exposes it as a Deep Agents backend.

Since Microsandbox provides an asynchronous API, sandbox creation and adapter initialization must be performed asynchronously.

### Basic example

```python
from microsandbox import Sandbox
from langchain_backend_microsandbox import MicrosandboxSandbox
from deepagents import create_deep_agent

model = init_chat_model("YOUR_MODEL")


async def main():
    # Create a sandbox from MicroSandbox API
    sandbox = await Sandbox.create(
        "deep-agent",
        image="python:3.13",
        memory=1024,
        cpus=2,
        ephemeral=True,
    )

    # Create a DeepAgent MicroSandbox Backend
    backend = await MicrosandboxSandbox.create(
        sandbox,
        timeout=1800,
    )

    try:
        agent = create_deep_agent(model=model, backend=backend)
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Write 'Hi !' in the /tmp/test.txt",
                    }
                ]
            },
        )
    finally:
        content_file = await backend.adownload_files(["/tmp/test.txt"])
        print(content_file)
        # [FileDownloadResponse(path='/tmp/test', content=b'Hi !', error=None)]

        # Sandbox lifecycle management remains the caller's responsibility.
        await sandbox.stop()
```

### Sandbox lifecyle

**The adapter does not create, start, stop, or destroy virtual machines.**

The calling application remains responsible for:

* Creating and configuring the Microsandbox instance.
* Managing its lifecycle.
* Defining its persistence policy.
* Stopping and cleaning up the sandbox when execution is complete.

Setting `ephemeral=True` when creating the sandbox allows Microsandbox to apply its ephemeral VM lifecycle policy (destroyed after stopping).

The adapter wraps an existing Microsandbox instance. Sandbox creation, lifecycle management and persistence remain the responsibility of the calling application.

> [!WARNING]
> The Microsandbox VM must have **Python 3 installed and available in PATH**. Without it, inherited `BaseSandbox` filesystem operations may fail.

## Configuration

`MicrosandboxSandbox.create()` accepts the following parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sandbox` | `Sandbox` | Required | Existing Microsandbox instance. |
| `timeout` | `int` | `1800` | Default command execution timeout in seconds. Use `0` to disable it. Must be positive. |
| `enable_capture_offload` | `bool` | `False` | Enables capture-at-source offloading for large command outputs. Enabling it requires a compatible shell environment and the utilities expected by Deep Agents' capture wrapper. |
| `grep_exclude_dirs` | `tuple[str, ...]` | `(".git",)` | Directory names excluded from recursive grep searches. |

### Example

```python
backend = await MicrosandboxSandbox.create(
    sandbox,
    timeout=600,
    enable_capture_offload=False,
    grep_exclude_dirs=(".git", "__pycache__", ".venv"),
)
```

## Testing

Tests use a real Microsandbox VM.

**A working Microsandbox runtime and hardware virtualization support are required.**

### Install development dependencies:

```bash
uv sync --group dev
```

### Run the test suite:

```bash
uv run pytest
```

### Run with verbose output and coverage

```bash
uv run pytest -v -s --tb=short --cov --cov-report=term-missing
```

## License

This project is distributed under the MIT License.

## Acknowledgements

This project builds upon the `BaseSandbox` interface provided by
[LangChain Deep Agents](https://github.com/langchain-ai/deepagents).

It is inspired by `kenwoodjw/langchain-microsandbox`, whose work helped establish the initial integration approach.

Deep Agents is distributed under the MIT License.

`langchain-backend-microsandbox` is an **independent community integration** and is not affiliated with, endorsed by, or officially supported by LangChain.
