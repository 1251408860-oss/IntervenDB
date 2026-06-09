from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "downloads"
LOG_DIR = DOWNLOAD_DIR / "logs"
STATE_PATH = DOWNLOAD_DIR / "dataset_download_state.json"


AUTO_FILES = [
    {
        "id": "uwf_benign",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/benign.csv",
        "path": "data/raw/uwf_zeekdata24/benign.csv",
        "size": 10163424,
        "md5": None,
    },
    {
        "id": "uwf_credential_access",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/credential_access.csv",
        "path": "data/raw/uwf_zeekdata24/credential_access.csv",
        "size": 10794836,
        "md5": None,
    },
    {
        "id": "uwf_defense_evasion",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/defense_evasion.csv",
        "path": "data/raw/uwf_zeekdata24/defense_evasion.csv",
        "size": 74867,
        "md5": None,
    },
    {
        "id": "uwf_exfiltration",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/exfiltration.csv",
        "path": "data/raw/uwf_zeekdata24/exfiltration.csv",
        "size": 6223,
        "md5": None,
    },
    {
        "id": "uwf_initial_access",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/initial_access.csv",
        "path": "data/raw/uwf_zeekdata24/initial_access.csv",
        "size": 135532,
        "md5": None,
    },
    {
        "id": "uwf_persistence",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/persistence.csv",
        "path": "data/raw/uwf_zeekdata24/persistence.csv",
        "size": 78127,
        "md5": None,
    },
    {
        "id": "uwf_privilege_escalation",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/privilege_escalation.csv",
        "path": "data/raw/uwf_zeekdata24/privilege_escalation.csv",
        "size": 81061,
        "md5": None,
    },
    {
        "id": "uwf_reconnaissance",
        "dataset": "UWF-ZeekData24",
        "groups": ["uwf", "core", "all_auto"],
        "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/csv/reconnaissance.csv",
        "path": "data/raw/uwf_zeekdata24/reconnaissance.csv",
        "size": 626614,
        "md5": None,
    },
    {
        "id": "gotham_2025_full_zip",
        "dataset": "Gotham Dataset 2025",
        "groups": ["gotham", "core", "all_auto"],
        "url": "https://zenodo.org/api/records/14502760/files/GothamDataset2025.zip/content",
        "path": "data/raw/gotham_2025/archive/GothamDataset2025.zip",
        "size": 23824968355,
        "md5": "7ca78c0517ccb3d2854e823678e0f206",
    },
    {
        "id": "cesnet_ip_addresses_sample",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_small", "cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/ip_addresses_sample.tar.gz/content",
        "path": "data/raw/cesnet_timeseries24/archive/ip_addresses_sample.tar.gz",
        "size": 170864480,
        "md5": "08451dab2d1eddb29a79467b03289232",
    },
    {
        "id": "cesnet_ids_relationship",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_small", "cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/ids_relationship.csv/content",
        "path": "data/raw/cesnet_timeseries24/archive/ids_relationship.csv",
        "size": 3824315,
        "md5": "f034e4e7f1844e36fa7d3a4a0cbfc600",
    },
    {
        "id": "cesnet_weekends_and_holidays",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_small", "cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/weekends_and_holidays.csv/content",
        "path": "data/raw/cesnet_timeseries24/archive/weekends_and_holidays.csv",
        "size": 1831,
        "md5": "f8578ebd8a1bc95a7187fed2e436e5bd",
    },
    {
        "id": "cesnet_institution_subnets",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/institution_subnets.tar.gz/content",
        "path": "data/raw/cesnet_timeseries24/archive/institution_subnets.tar.gz",
        "size": 774004934,
        "md5": "735b5ff436b6c67556cb67e2888b14ea",
    },
    {
        "id": "cesnet_ip_addresses_full",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/ip_addresses_full.tar.gz/content",
        "path": "data/raw/cesnet_timeseries24/archive/ip_addresses_full.tar.gz",
        "size": 40048881181,
        "md5": "7c3d28b7b2d2a02430ee88f84fb823b6",
    },
    {
        "id": "cesnet_institutions",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/institutions.tar.gz/content",
        "path": "data/raw/cesnet_timeseries24/archive/institutions.tar.gz",
        "size": 479428489,
        "md5": "ab3e15fb8dc9b7120ddb2318795b6812",
    },
    {
        "id": "cesnet_times",
        "dataset": "CESNET-TimeSeries24",
        "groups": ["cesnet_full", "all_auto"],
        "url": "https://zenodo.org/api/records/13382427/files/times.tar.gz/content",
        "path": "data/raw/cesnet_timeseries24/archive/times.tar.gz",
        "size": 211467,
        "md5": "a03813763e07646ca38f17ffd53e549e",
    },
]


MANUAL_DATASETS = [
    {
        "id": "datasense_iiot_2025",
        "dataset": "DataSense CIC IIoT Dataset 2025",
        "source": "https://www.unb.ca/cic/datasets/iiot-dataset-2025.html",
        "registration": "https://cicresearch.ca/IOTDataset/Datasense/",
        "reason": "CIC registration form required; no public file URL is exposed before registration.",
    },
    {
        "id": "cicapt_iiot_2024",
        "dataset": "CICAPT-IIoT2024",
        "source": "https://www.unb.ca/cic/datasets/iiot-dataset-2024.html",
        "registration": "https://cicresearch.ca/IOTDataset/CICAPT-IIoT-Dataset/",
        "reason": "CIC registration form required; no public file URL is exposed before registration.",
    },
    {
        "id": "bccc_aposemat_iot_bot_2024",
        "dataset": "BCCC-Aposemat-IoT-BoT-2024",
        "source": "https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/bccc-aposemat-bot-iot-2024-developed-and-designed-for-large-language-models-llm/",
        "registration": None,
        "reason": "Manual source inspection is needed before adding a stable direct-download URL.",
    },
    {
        "id": "flnet2023",
        "dataset": "FLNET2023",
        "source": "https://github.com/nsol-nmsu/FML-Network",
        "registration": None,
        "reason": "Repository is available, but dataset files are not in the current direct-download manifest.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def target_path(item: dict) -> Path:
    return ROOT / item["path"]


def selected_items(group: str, only: list[str] | None) -> list[dict]:
    if only:
        ids = set(only)
        items = [item for item in AUTO_FILES if item["id"] in ids]
        missing = sorted(ids - {item["id"] for item in items})
        if missing:
            raise SystemExit(f"Unknown dataset ids: {', '.join(missing)}")
        return items
    return [item for item in AUTO_FILES if group in item["groups"]]


def file_state(item: dict) -> dict:
    path = target_path(item)
    expected = item.get("size")
    size = path.stat().st_size if path.exists() else 0
    if expected is not None and size == expected:
        status = "complete"
    elif size > 0:
        status = "partial"
    else:
        status = "missing"
    if expected is not None and size > expected:
        status = "size_mismatch"
    pct = None if not expected else min(size / expected * 100.0, 999.0)
    return {
        "id": item["id"],
        "dataset": item["dataset"],
        "path": rel_path(path),
        "status": status,
        "local_size": size,
        "expected_size": expected,
        "percent": pct,
    }


def print_status(items: list[dict]) -> None:
    usage = shutil.disk_usage(ROOT.anchor or ROOT)
    print(f"Project root: {ROOT}")
    print(f"Disk free: {human_size(usage.free)} / total {human_size(usage.total)}")
    print("")
    print(f"{'status':<14} {'percent':>8} {'local':>12} {'expected':>12}  id")
    print("-" * 78)
    for item in items:
        state = file_state(item)
        pct = "-" if state["percent"] is None else f"{state['percent']:.2f}%"
        print(
            f"{state['status']:<14} {pct:>8} "
            f"{human_size(state['local_size']):>12} "
            f"{human_size(state['expected_size']):>12}  {state['id']}"
        )
    print("")
    print("Manual or gated datasets:")
    for item in MANUAL_DATASETS:
        reg = item["registration"] or item["source"]
        print(f"- {item['id']}: {item['dataset']} ({reg})")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024 * 8), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(item: dict, verify_md5: bool) -> bool:
    path = target_path(item)
    state = file_state(item)
    if state["status"] != "complete":
        print(f"not complete: {item['id']} ({state['status']})")
        return False
    expected_md5 = item.get("md5")
    if verify_md5 and expected_md5:
        actual = md5sum(path)
        if actual.lower() != expected_md5.lower():
            print(f"md5 mismatch: {item['id']} expected={expected_md5} actual={actual}")
            return False
        print(f"verified md5: {item['id']}")
    else:
        print(f"verified size: {item['id']}")
    return True


def curl_path() -> str:
    path = shutil.which("curl.exe") or shutil.which("curl")
    if not path:
        raise SystemExit("curl was not found in PATH")
    return path


def curl_proxy_args(url: str) -> list[str]:
    explicit = os.environ.get("DATASET_PROXY")
    if explicit:
        return ["--proxy", explicit]
    proxies = urllib.request.getproxies()
    proxy = proxies.get("https" if url.lower().startswith("https:") else "http")
    return ["--proxy", proxy] if proxy else []


def download_one(item: dict) -> bool:
    path = target_path(item)
    state = file_state(item)
    if state["status"] == "complete":
        print(f"skip complete: {item['id']}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{item['id']}.log"
    cmd = [
        curl_path(),
        "-L",
        "-C",
        "-",
        "--fail",
        "--retry",
        "30",
        "--retry-delay",
        "10",
        "--connect-timeout",
        "30",
        "--speed-time",
        "180",
        "--speed-limit",
        "1024",
        "-o",
        str(path),
    ]
    cmd.extend(curl_proxy_args(item["url"]))
    cmd.append(item["url"])

    print(f"download: {item['id']} -> {rel_path(path)}")
    print(f"log: {rel_path(log_path)}")
    with log_path.open("ab") as log:
        log.write(f"\n\n[{utc_now()}] start {' '.join(cmd)}\n".encode("utf-8"))
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[{utc_now()}] exit_code={proc.returncode}\n".encode("utf-8"))

    state_after = file_state(item)
    if proc.returncode == 0 and state_after["status"] == "complete":
        print(f"complete: {item['id']}")
        return True
    print(f"incomplete: {item['id']} status={state_after['status']}")
    return False


def run_downloads(items: list[dict], concurrency: int) -> int:
    if concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if concurrency == 1:
        ok = [download_one(item) for item in items]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            ok = list(pool.map(download_one, items))
    return 0 if all(ok) else 1


def parse_part_start(path: Path) -> int | None:
    try:
        return int(path.name.split("-", 1)[0])
    except (ValueError, IndexError):
        return None


def download_range_part(item: dict, start: int, end: int, seg_dir: Path, log_path: Path, log_lock: threading.Lock) -> Path:
    span = end - start + 1
    final = seg_dir / f"{start:014d}-{end:014d}.part"
    tmp = seg_dir / f"{start:014d}-{end:014d}.tmp"
    if final.exists() and final.stat().st_size == span:
        return final
    for path in [final, tmp]:
        if path.exists():
            path.unlink()

    cmd = [
        curl_path(),
        "-sS",
        "-L",
        "--fail",
        "--retry",
        "30",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "30",
        "--speed-time",
        "180",
        "--speed-limit",
        "1024",
        "-r",
        f"{start}-{end}",
        "-o",
        str(tmp),
    ]
    cmd.extend(curl_proxy_args(item["url"]))
    cmd.append(item["url"])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = proc.stdout.decode("utf-8", errors="replace")
    with log_lock:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{utc_now()}] range {start}-{end} exit={proc.returncode}\n")
            if output:
                log.write(output)
                if not output.endswith("\n"):
                    log.write("\n")
    if proc.returncode != 0:
        raise RuntimeError(f"range download failed for {item['id']} {start}-{end}")
    if not tmp.exists() or tmp.stat().st_size != span:
        actual = tmp.stat().st_size if tmp.exists() else 0
        raise RuntimeError(f"range size mismatch for {item['id']} {start}-{end}: {actual} != {span}")
    tmp.replace(final)
    return final


def segment_download_one(item: dict, workers: int, chunk_mib: int) -> bool:
    expected = item.get("size")
    if not expected:
        print(f"skip without expected size: {item['id']}")
        return False
    path = target_path(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()

    current = path.stat().st_size
    if current == expected:
        print(f"skip complete: {item['id']}")
        return True
    if current > expected:
        print(f"size mismatch: {item['id']} local={current} expected={expected}")
        return False

    chunk = chunk_mib * 1024 * 1024
    seg_dir = DOWNLOAD_DIR / "segments" / item["id"]
    seg_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"segmented_{item['id']}.log"

    for part in seg_dir.glob("*"):
        start = parse_part_start(part)
        if start is None or start < current:
            part.unlink()

    ranges: list[tuple[int, int]] = []
    start = current
    while start < expected:
        end = min(start + chunk - 1, expected - 1)
        ranges.append((start, end))
        start = end + 1

    print(
        f"segment-download: {item['id']} start={human_size(current)} "
        f"expected={human_size(expected)} ranges={len(ranges)} workers={workers}"
    )
    if not ranges:
        return path.stat().st_size == expected

    log_lock = threading.Lock()
    completed: dict[int, Path] = {}
    futures: dict[concurrent.futures.Future[Path], int] = {}
    submit_index = 0
    append_index = 0
    appended_since_report = 0
    append_pos = current

    max_buffered = workers * 2

    def submit(pool: concurrent.futures.ThreadPoolExecutor) -> None:
        nonlocal submit_index
        while submit_index < len(ranges) and len(futures) + len(completed) < max_buffered:
            range_start, range_end = ranges[submit_index]
            fut = pool.submit(download_range_part, item, range_start, range_end, seg_dir, log_path, log_lock)
            futures[fut] = submit_index
            submit_index += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool, path.open("ab") as out:
        submit(pool)
        while append_index < len(ranges):
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in done:
                idx = futures.pop(fut)
                completed[idx] = fut.result()
            submit(pool)
            while append_index in completed:
                part = completed.pop(append_index)
                range_start, range_end = ranges[append_index]
                if range_start != append_pos:
                    raise RuntimeError(f"append position mismatch for {item['id']}: {range_start} != {append_pos}")
                with part.open("rb") as src:
                    shutil.copyfileobj(src, out, length=1024 * 1024 * 8)
                out.flush()
                part.unlink()
                append_pos = range_end + 1
                appended_since_report += range_end - range_start + 1
                if appended_since_report >= 1024 * 1024 * 1024 or append_pos == expected:
                    pct = append_pos / expected * 100.0
                    print(f"{item['id']}: {human_size(append_pos)} / {human_size(expected)} ({pct:.2f}%)")
                    appended_since_report = 0
                append_index += 1
                submit(pool)

    final_size = path.stat().st_size
    if final_size != expected:
        print(f"incomplete segmented download: {item['id']} {final_size} != {expected}")
        return False
    print(f"complete: {item['id']}")
    return True


def run_segment_downloads(items: list[dict], workers: int, chunk_mib: int) -> int:
    if workers < 1:
        raise SystemExit("--workers must be >= 1")
    if chunk_mib < 8:
        raise SystemExit("--chunk-mib must be >= 8")
    ok = [segment_download_one(item, workers, chunk_mib) for item in items]
    return 0 if all(ok) else 1


def launch_background(args: argparse.Namespace) -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"background_{args.group}_{stamp}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "download",
        "--group",
        args.group,
        "--concurrency",
        str(args.concurrency),
    ]
    for item_id in args.only or []:
        cmd.extend(["--only", item_id])

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    STATE_PATH.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "started_at": utc_now(),
                "group": args.group,
                "concurrency": args.concurrency,
                "command": cmd,
                "log": rel_path(log_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"started background download pid={proc.pid}")
    print(f"log: {rel_path(log_path)}")
    print(f"state: {rel_path(STATE_PATH)}")


def extract_gotham_processed() -> int:
    archive = ROOT / "data/raw/gotham_2025/archive/GothamDataset2025.zip"
    out_dir = ROOT / "data/raw/gotham_2025/processed_full"
    if not archive.exists():
        print(f"missing archive: {rel_path(archive)}")
        return 1
    expected = next(item["size"] for item in AUTO_FILES if item["id"] == "gotham_2025_full_zip")
    if archive.stat().st_size != expected:
        print(f"archive is not complete: {human_size(archive.stat().st_size)} / {human_size(expected)}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        members = [
            member
            for member in zf.namelist()
            if member.startswith("processed/") and member.lower().endswith(".csv")
        ]
        for member in members:
            target = out_dir / Path(member).name
            if target.exists() and target.stat().st_size > 0:
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 8)
            count += 1
    print(f"processed csv dir: {rel_path(out_dir)}")
    print(f"newly extracted files: {count}")
    return 0


def write_manifest() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": str(ROOT),
        "updated_at": utc_now(),
        "auto_files": AUTO_FILES,
        "manual_datasets": MANUAL_DATASETS,
    }
    path = DOWNLOAD_DIR / "dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {rel_path(path)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage IntervenDB dataset downloads.")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ["status", "download", "verify", "background"]:
        p = sub.add_parser(name)
        p.add_argument(
            "--group",
            default="all_auto",
            choices=["uwf", "gotham", "core", "cesnet_small", "cesnet_full", "all_auto"],
        )
        p.add_argument("--only", action="append", default=[])
        if name in {"download", "background"}:
            p.add_argument("--concurrency", type=int, default=1)
        if name == "verify":
            p.add_argument("--verify-md5", action="store_true")

    p = sub.add_parser("segment-download")
    p.add_argument("--group", default="all_auto", choices=["uwf", "gotham", "core", "cesnet_small", "cesnet_full", "all_auto"])
    p.add_argument("--only", action="append", default=[])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-mib", type=int, default=128)

    sub.add_parser("extract-gotham-processed")
    sub.add_parser("write-manifest")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "extract-gotham-processed":
        return extract_gotham_processed()
    if args.command == "write-manifest":
        write_manifest()
        return 0

    items = selected_items(args.group, args.only)
    if args.command == "status":
        print_status(items)
        return 0
    if args.command == "download":
        return run_downloads(items, args.concurrency)
    if args.command == "verify":
        ok = [verify_file(item, args.verify_md5) for item in items]
        return 0 if all(ok) else 1
    if args.command == "background":
        launch_background(args)
        return 0
    if args.command == "segment-download":
        return run_segment_downloads(items, args.workers, args.chunk_mib)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


