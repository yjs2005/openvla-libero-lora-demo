#!/usr/bin/env python
"""Generate a lightweight Markdown environment report for this project."""

from __future__ import annotations

import ctypes
import csv
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "notes" / "environment_report.md"


@dataclass
class CommandResult:
    ok: bool
    output: str
    error: str = ""


def run_command(args: Iterable[str], timeout: int = 15) -> CommandResult:
    try:
        completed = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return CommandResult(False, "", f"command not found: {exc}")
    except subprocess.TimeoutExpired as exc:
        return CommandResult(False, exc.stdout or "", f"timeout: {exc}")
    except Exception as exc:  # pragma: no cover - defensive reporting
        return CommandResult(False, "", f"{type(exc).__name__}: {exc}")

    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    return CommandResult(completed.returncode == 0, output, error)


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "Unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def format_mib(num_mib: int | float | None) -> str:
    if num_mib is None:
        return "Unknown"
    return f"{num_mib:.0f} MiB ({num_mib / 1024:.2f} GB)"


def get_windows_memory() -> tuple[int | None, int | None]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    return None, None


def get_linux_memory() -> tuple[int | None, int | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        match = re.search(r"(\d+)", raw_value)
        if match:
            values[key] = int(match.group(1)) * 1024
    return values.get("MemTotal"), values.get("MemAvailable") or values.get("MemFree")


def get_macos_memory() -> tuple[int | None, int | None]:
    total = None
    available = None
    total_result = run_command(["sysctl", "-n", "hw.memsize"])
    if total_result.ok and total_result.output.strip().isdigit():
        total = int(total_result.output.strip())

    vm_result = run_command(["vm_stat"])
    page_size = 4096
    if vm_result.ok:
        page_match = re.search(r"page size of (\d+) bytes", vm_result.output)
        if page_match:
            page_size = int(page_match.group(1))
        free_pages = 0
        for line in vm_result.output.splitlines():
            if line.startswith(("Pages free", "Pages inactive", "Pages speculative")):
                count_match = re.search(r"(\d+)", line.replace(".", ""))
                if count_match:
                    free_pages += int(count_match.group(1))
        available = free_pages * page_size if free_pages else None

    return total, available


def get_memory() -> tuple[int | None, int | None]:
    system = platform.system().lower()
    if system == "windows":
        return get_windows_memory()
    if system == "linux":
        return get_linux_memory()
    if system == "darwin":
        return get_macos_memory()
    return None, None


def get_cpu_name() -> str:
    system = platform.system().lower()
    if system == "windows":
        result = run_command(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
            ]
        )
        if result.ok and result.output:
            return result.output.strip()
    if system == "linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if system == "darwin":
        result = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if result.ok and result.output:
            return result.output.strip()
    return platform.processor() or "Unknown"


def get_tool_version(command: str, args: list[str]) -> tuple[bool, str]:
    if shutil.which(command) is None:
        return False, "Not found"
    result = run_command([command, *args])
    if result.ok:
        return True, result.output.splitlines()[0] if result.output else "Available"
    detail = result.error or result.output or "Failed"
    return False, detail


def parse_nvidia_smi() -> tuple[bool, list[dict[str, str]], str, str, str]:
    if shutil.which("nvidia-smi") is None:
        return False, [], "Not found", "Unknown", "Unknown"

    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    raw = run_command(["nvidia-smi"])

    cuda_version = "Unknown"
    if raw.output:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", raw.output)
        if match:
            cuda_version = match.group(1)

    gpus: list[dict[str, str]] = []
    if query.ok and query.output:
        reader = csv.reader(query.output.splitlines())
        for row in reader:
            if len(row) < 5:
                continue
            name, total, used, free, driver = [item.strip() for item in row[:5]]
            gpus.append(
                {
                    "name": name,
                    "memory_total_mib": total,
                    "memory_used_mib": used,
                    "memory_free_mib": free,
                    "driver_version": driver,
                }
            )

    status = "Available" if query.ok else "Failed"
    detail = query.error or query.output or "nvidia-smi returned no query output"
    return True, gpus, detail, cuda_version, raw.output or raw.error


def get_torch_info() -> dict[str, str]:
    info = {
        "installed": "No",
        "version": "N/A",
        "cuda_is_available": "N/A",
        "torch_cuda_version": "N/A",
        "gpu_name": "N/A",
        "cuda_mem_get_info": "N/A",
        "error": "",
    }

    if importlib.util.find_spec("torch") is None:
        return info

    try:
        import torch  # type: ignore

        info["installed"] = "Yes"
        info["version"] = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        info["cuda_is_available"] = str(cuda_available)
        info["torch_cuda_version"] = str(torch.version.cuda)
        if cuda_available:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            info["cuda_mem_get_info"] = (
                f"free {format_bytes(free_bytes)} / total {format_bytes(total_bytes)}"
            )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def as_int(value: str) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def build_conclusions(
    gpus: list[dict[str, str]],
    disk_free_bytes: int,
    torch_info: dict[str, str],
) -> list[str]:
    conclusions: list[str] = []
    max_vram_mib = 0
    for gpu in gpus:
        total = as_int(gpu.get("memory_total_mib", ""))
        if total:
            max_vram_mib = max(max_vram_mib, total)

    if not gpus:
        conclusions.append(
            "No NVIDIA GPU detected: only suitable for code reading, data processing, README work, and light CPU tests; not suitable for OpenVLA inference or fine-tuning."
        )
    else:
        vram_gb = max_vram_mib / 1024
        if vram_gb < 16:
            conclusions.append(
                "GPU memory is below 16GB: not recommended for OpenVLA-7B; at most use this machine for environment preparation and very light code tests."
            )
        elif 16 <= vram_gb < 24:
            conclusions.append(
                "GPU memory is 16GB to 24GB: small inference or extremely small-batch experiments may work, but LoRA fine-tuning is risky and should use batch_size=1, gradient accumulation, bf16/fp16, and very small data."
            )
        elif 24 <= vram_gb < 40:
            conclusions.append(
                "GPU memory is 24GB to 40GB: small-scale OpenVLA LoRA fine-tuning can be attempted with controlled batch size and training steps."
            )
        else:
            conclusions.append(
                "GPU memory is at least 40GB: comparatively suitable for OpenVLA inference evaluation and small-scale LoRA fine-tuning."
            )

    disk_free_gb = disk_free_bytes / (1024**3)
    if disk_free_gb < 30:
        conclusions.append("Disk free space is below 30GB: not recommended for downloading LIBERO data or model checkpoints.")
    elif 30 <= disk_free_gb < 100:
        conclusions.append("Disk free space is 30GB to 100GB: can try small LIBERO data and checkpoints, but not suitable for BridgeData V2.")
    elif disk_free_gb > 150:
        conclusions.append("Disk free space is above 150GB: storage is relatively sufficient for LIBERO and some later BridgeData V2 work.")
    else:
        conclusions.append("Disk free space is 100GB to 150GB: enough for controlled LIBERO experiments, but large checkpoints and BridgeData V2 still require caution.")

    cuda_ready = bool(gpus) and torch_info.get("cuda_is_available") == "True"
    if not gpus:
        conclusions.extend(
            [
                "OpenVLA/LIBERO inference evaluation: Not suitable on this machine without an NVIDIA GPU.",
                "Small-scale LoRA fine-tuning: Not suitable on this machine without an NVIDIA GPU.",
                "Full large-scale training: Not suitable.",
            ]
        )
    elif max_vram_mib < 16 * 1024:
        conclusions.extend(
            [
                "OpenVLA/LIBERO inference evaluation: Not recommended for OpenVLA-7B on this GPU memory size.",
                "Small-scale LoRA fine-tuning: Not recommended on this GPU memory size.",
                "Full large-scale training: Not suitable.",
            ]
        )
    elif max_vram_mib < 24 * 1024:
        conclusions.extend(
            [
                "OpenVLA/LIBERO inference evaluation: Possibly suitable only for small debug runs with conservative memory settings.",
                "Small-scale LoRA fine-tuning: High risk; only attempt tiny experiments after confirming official memory requirements.",
                "Full large-scale training: Not suitable.",
            ]
        )
    elif max_vram_mib < 40 * 1024:
        conclusions.extend(
            [
                "OpenVLA/LIBERO inference evaluation: Suitable to try with controlled settings.",
                "Small-scale LoRA fine-tuning: Suitable to try with small batches and short runs.",
                "Full large-scale training: Not suitable.",
            ]
        )
    else:
        conclusions.extend(
            [
                "OpenVLA/LIBERO inference evaluation: Suitable.",
                "Small-scale LoRA fine-tuning: Suitable for controlled experiments.",
                "Full large-scale training: Still not assumed suitable without multi-GPU/storage planning.",
            ]
        )

    if gpus and not cuda_ready:
        conclusions.append(
            "PyTorch CUDA is not currently available, so GPU workloads may require installing a CUDA-enabled PyTorch build before running OpenVLA."
        )

    return conclusions


def markdown_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Item | Value |", "| --- | --- |"]
    for key, value in rows:
        safe_value = str(value).replace("\n", "<br>")
        lines.append(f"| {key} | {safe_value} |")
    return "\n".join(lines)


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_mem, available_mem = get_memory()
    disk = shutil.disk_usage(PROJECT_ROOT)
    git_available, git_version = get_tool_version("git", ["--version"])
    gh_available, gh_version = get_tool_version("gh", ["--version"])
    nvidia_present, gpus, nvidia_detail, nvidia_cuda_version, nvidia_raw = parse_nvidia_smi()
    torch_info = get_torch_info()

    os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
    python_version = sys.version.replace("\n", " ")
    cpu_name = get_cpu_name()

    lines: list[str] = []
    lines.append("# Environment Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Project root: `{PROJECT_ROOT}`")
    lines.append("")

    lines.append("## System")
    lines.append("")
    lines.append(
        markdown_table(
            [
                ("OS", os_name),
                ("Python version", python_version),
                ("CPU", cpu_name),
                ("CPU cores / logical CPUs", f"{os.cpu_count()} logical CPUs"),
                ("RAM total", format_bytes(total_mem)),
                ("RAM available", format_bytes(available_mem)),
                ("Disk total", format_bytes(disk.total)),
                ("Disk free", format_bytes(disk.free)),
            ]
        )
    )
    lines.append("")

    lines.append("## CLI Tools")
    lines.append("")
    lines.append(
        markdown_table(
            [
                ("git available", str(git_available)),
                ("git version", git_version),
                ("gh available", str(gh_available)),
                ("gh version", gh_version),
            ]
        )
    )
    lines.append("")

    lines.append("## NVIDIA GPU")
    lines.append("")
    lines.append(
        markdown_table(
            [
                ("nvidia-smi present", str(nvidia_present)),
                ("nvidia-smi query status", nvidia_detail if not gpus else "OK"),
                ("CUDA version reported by nvidia-smi", nvidia_cuda_version),
            ]
        )
    )
    lines.append("")

    if gpus:
        lines.append("| GPU | Name | Memory total | Memory used | Memory free | Driver version |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for idx, gpu in enumerate(gpus):
            total_mib = as_int(gpu["memory_total_mib"])
            used_mib = as_int(gpu["memory_used_mib"])
            free_mib = as_int(gpu["memory_free_mib"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        gpu["name"],
                        format_mib(total_mib),
                        format_mib(used_mib),
                        format_mib(free_mib),
                        gpu["driver_version"],
                    ]
                )
                + " |"
            )
    else:
        lines.append("No NVIDIA GPU details were detected.")
    lines.append("")

    if nvidia_raw:
        lines.append("<details>")
        lines.append("<summary>Raw nvidia-smi output</summary>")
        lines.append("")
        lines.append("```text")
        lines.append(nvidia_raw)
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("## PyTorch")
    lines.append("")
    lines.append(
        markdown_table(
            [
                ("PyTorch installed", torch_info["installed"]),
                ("torch version", torch_info["version"]),
                ("torch.cuda.is_available()", torch_info["cuda_is_available"]),
                ("torch.version.cuda", torch_info["torch_cuda_version"]),
                ("torch.cuda.get_device_name()", torch_info["gpu_name"]),
                ("torch.cuda.mem_get_info()", torch_info["cuda_mem_get_info"]),
                ("PyTorch detection error", torch_info["error"] or "None"),
            ]
        )
    )
    lines.append("")

    lines.append("## Suitability Assessment")
    lines.append("")
    for conclusion in build_conclusions(gpus, disk.free, torch_info):
        lines.append(f"- {conclusion}")
    lines.append("")

    report = "\n".join(lines)
    # Use UTF-8 with BOM so Windows PowerShell and Notepad display Chinese paths correctly.
    REPORT_PATH.write_text(report, encoding="utf-8-sig")
    print(report)
    print(f"\nReport saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
