# ----------------------------------------------------------------------
# Project: Right In-Place Convolution
# Title:   mcunet_comparison_results.py
#
# Reference papers:
#    Yet to be published
# Contact authors:
#  - Opegbemi Matthias Busoye, matthias@powerlabstech.com
#  - Tolulope Matthew Busoye, matthew@powerlabstech.com
#  - Eghonghon-aye Eigbe, eghonghon@powerlabstech.com
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

"""
Whole-model sweep over mcunet-in0 .. mcunet-in4
============================================================

Measures right in-place convolution against an unmodified TinyEngine, one
whole model at a time, and records both sides of every number.

Per model
---------
    1. run codegen in the baseline tree      -> baseline_peak_mem
    2. run codegen here                      -> peak_mem
    3. with -b: build, flash and read the baseline firmware
    4. with -b: build, flash and read this tree's firmware

Both trees run the same examples/mcunet_inx.py, the same deploy/pc harness
and the same Pico main.c, on the same board, back to back. The only thing
that differs is the memory scheduler, which is what makes the comparison
meaningful.

Usage
-----
    python mcunet_comparison_results.py                     # codegen only
    python mcunet_comparison_results.py -b pico2 -f         # + both firmwares
    python mcunet_comparison_results.py -x 0 1 -b pico2 -f  # a subset
    python mcunet_comparison_results.py -c "" -f            # this tree alone

Run from the repository root.

Writes
------
    ../results/mcunet_comparison.csv          one row per model
    ../results/mcunet_summary_comparison.csv  mean/min/max/median/std
    ../results/mcunet_comparison_results.log

Reading the results
-------------------
peak_mem_ratio below 1 is a saving. checksums_agree is the correctness
check: the two builds must produce the same output, and a mismatch is
logged as an error rather than quietly recorded. cycles and duration_us
come from the DWT cycle counter and time_us_64 respectively, so they
measure the same work two ways.

A model is skipped, with a warning, when its peak exceeds the board's
usable SRAM (mcunet-in4 does not fit an RP2040). Rows are flushed as they
are produced, so a board that stops responding halfway through does not
cost the models already measured.
"""

import os
import re
import sys
import csv
import time
import logging
import argparse
import subprocess

import serial
import numpy as np

RESULTS_FILE = "../results/mcunet_comparison.csv"
SUMMARY_RESULTS_FILE = "../results/mcunet_summary_comparison.csv"
LOG_FILE = "../results/mcunet_comparison_results.log"

PICO_SOURCE_DIR = {".": "deploy_pico2"}
DEFAULT_PICO_SOURCE_DIR = "deploy/pico2"
DEFAULT_SDK_PATH = os.path.expanduser("~/.pico-sdk/sdk/2.2.0")

# Usable SRAM per board, less a margin for the SDK, stacks and USB buffers.
# A model whose peak exceeds this is skipped rather than left to fail at link.
RAM_SIZE_MARGIN = 20 * 1024
BOARD_SRAM = {
    "pico": 264 * 1024,   # RP2040, Cortex-M0+
    "pico2": 520 * 1024,  # RP2350, Cortex-M33
}

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser("mcunet model results")
    parser.add_argument(
        "-x", "--models", required=False, type=int, nargs="+", default=list(range(0, 5)),
        choices=list(range(0, 5)),
        help="Which mcunet-inX models to sweep. Defaults to all five."
    )

    parser.add_argument(
        "-b", "--pico_board", required=False, choices=["pico", "pico2"], default=None,
        help="What board would the experiment be run on. Omit for codegen only."
    )

    parser.add_argument(
        "-c", "--baseline_comparison", required=False, type=str, default="../tinyengine",
        help="Unmodified TinyEngine checkout to benchmark against. Pass an empty "
            "string to skip the baseline and report this tree's numbers alone."
    )

    parser.add_argument(
        "-t", "--timeout", required=False, type=int, default=None,
        help="The max time each on-device run should take."
    )

    parser.add_argument(
        "-p", "--port", required=False, type=str, default="/dev/ttyACM0",
        help="The port the board is connected to read or load data for the experiment."
    )

    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Set it if you want to force the experiment to rerun if the results file exist already."
    )

    return parser.parse_args()

def cmd_runner(cmd, cwd="."):
    return subprocess.run(
        cmd, capture_output=True, check=True, shell=True, text=True,
        cwd=cwd, executable="/usr/bin/bash"
    )


def print_subprocess_error_message(error):
    cmd = error.cmd if isinstance(error.cmd, str) else " ".join(error.cmd)
    messages = [f"command failed (exit code {error.returncode})", f"  $ {cmd}"]
    if error.stdout and error.stdout.strip():
        messages.append(f"--- stdout ---\n{error.stdout.strip()}")
    if error.stderr and error.stderr.strip():
        messages.append(f"--- stderr ---\n{error.stderr.strip()}")
    logger.error("\n".join(messages))


def generate_model(x, cwd="."):
    """
    Run codegen for mcunet-inx in `cwd`. Returns the peak activation memory
    in bytes. Same script in both trees, so only the scheduler differs.
    """
    result = cmd_runner(f"python examples/mcunet_inx.py -x {x}", cwd=cwd)
    match = re.search(r"Peak memory:\s*(\d+)\s*bytes", result.stdout)
    assert match, f"codegen did not report a peak memory for in{x} in {cwd}:\n{result.stdout[-2000:]}"
    return int(match.group(1))


def read_model_shape():
    """Input / output dimensions of the model currently in codegen/."""
    with open("codegen/Include/genModelShape.h") as file:
        text = file.read()

    def _define(name):
        match = re.search(rf"#define {name}\s+(\d+)", text)
        return int(match.group(1)) if match else None

    return {
        "input_h": _define("INPUT_H"), "input_w": _define("INPUT_W"),
        "input_c": _define("INPUT_C"), "output_c": _define("OUTPUT_C"),
    }


def build_firmware(pico_board, cwd="."):
    """
    Build the Pico firmware for whatever `cwd` last generated. Returns the
    uf2 path, relative to `cwd`.
    """
    source_dir = PICO_SOURCE_DIR.get(cwd, DEFAULT_PICO_SOURCE_DIR)
    build_dir = os.path.join(source_dir, "build")
    sdk = os.environ.get("PICO_SDK_PATH", DEFAULT_SDK_PATH)
    cmd_runner(
        f"PICO_SDK_PATH={sdk} cmake -S {source_dir} -B {build_dir} -DPICO_BOARD={pico_board}",
        cwd=cwd,
    )
    cmd_runner(f"cmake --build {build_dir} -j$(nproc)", cwd=cwd)
    return os.path.join(build_dir, "blink.uf2")


def read_pico_serial(timeout=None, port="/dev/ttyACM0", baudrate=115200):
    """
    Captures the harness output over USB CDC after a `picotool load -x`
    reboot. deploy_pico2/main.c blocks on stdio_usb_connected(), so nothing
    is printed until this attaches; the checksum line is printed last and
    ends the capture.
    """
    for _ in range(50):
        if os.path.exists(port) and os.access(port, os.R_OK | os.W_OK):
            break
        time.sleep(0.1)
    else:
        raise TimeoutError(f"{port} never became accessible -- is the board connected and running?")

    # os.access above still races with udev's group/permission fixup on a
    # freshly re-enumerated device node, so retry the open itself for a bit
    # rather than treating one PermissionError as fatal.
    ser = None
    for _ in range(20):
        try:
            ser = serial.Serial(port, baudrate, timeout=0.5)
            break
        except (PermissionError, serial.SerialException):
            time.sleep(0.1)
    if ser is None:
        raise TimeoutError(f"{port} exists but never became openable -- udev permissions delayed?")

    deadline = time.time() + timeout if timeout is not None else float("inf")
    lines = []
    with ser:
        while time.time() < deadline:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace")
            lines.append(text)
            if re.search(r"checksum\s*=", text):
                break

    if not any(re.search(r"checksum\s*=", line) for line in lines):
        raise TimeoutError(
            f"timeout of {timeout}sec not enough to run the model, try a longer timeout "
            "or set timeout to None to wait until each run completes."
        )
    return "".join(lines)


def parse_pico_output(stdout):
    """
    Extracts the numbers deploy_pico2/main.c prints, by searching for each
    pattern anywhere in the text rather than by line position.

    main.c loops forever reprinting the block, so the capture stops at the
    first checksum line and only ever sees one complete set of numbers.
    """
    def _search(pattern, group=1):
        match = re.search(pattern, stdout)
        return match.group(group) if match else None

    cycles = _search(r"cycles\s*=\s*(\d+)")
    duration = _search(r"duration\s*=\s*(\d+)\s*us")
    stack = _search(r"stack used\s*=\s*(\d+)\s*bytes")
    checksum = _search(r"checksum\s*=\s*(-?\d+)")

    return {
        "cycles": int(cycles) if cycles else None,
        "duration_us": int(duration) if duration else None,
        "stack_used": int(stack) if stack else None,
        "checksum": int(checksum) if checksum is not None else None,
    }


def run_on_board(x, peak_mem, pico_board, timeout, port, cwd=".", label="right-inplace"):
    """
    Build, flash and read back one model from the tree at `cwd`. Returns None
    if it cannot be run, so the caller records the row either way.
    """
    usable = BOARD_SRAM[pico_board] - RAM_SIZE_MARGIN
    if peak_mem > usable:
        logger.warning(
            f"in{x} [{label}]: skipping {pico_board}, needs {peak_mem}B of activation "
            f"memory but only {usable}B is usable (after the {RAM_SIZE_MARGIN}B "
            "SDK/stack margin)"
        )
        return None

    try:
        uf2_path = build_firmware(pico_board, cwd=cwd)
    except subprocess.CalledProcessError as error:
        print_subprocess_error_message(error)
        return None

    try:
        # -f forces a running device to reboot into BOOTSEL first, so this works
        # whether or not the board is already there. -x runs it after flashing.
        cmd_runner(f"picotool load -f -x -v {uf2_path}", cwd=cwd)
    except subprocess.CalledProcessError as error:
        print_subprocess_error_message(error)
        return None

    try:
        stdout = read_pico_serial(timeout, port)
    except TimeoutError as error:
        logger.error(f"in{x} [{label}]: {error}")
        return None

    return parse_pico_output(stdout)


args = get_args()

if args.pico_board is not None:
    # Validating that a board is plugged based on the expected picotool message
    result = subprocess.run(["picotool", "info"], capture_output=True, text=True)
    no_bootsel_device = "No accessible RP-series devices in BOOTSEL mode were found." in result.stdout
    device_running = "appears to have a USB serial connection" in result.stdout
    assert (not no_bootsel_device) or device_running, \
        f"No pico board connected or picotool is not installed.\n{result.stdout}"

# -f/--force guards against silently clobbering a previous run's results
assert args.force or not os.path.exists(RESULTS_FILE), (
    f"{RESULTS_FILE} already exists -- pass -f/--force to overwrite it"
)

os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_RESULTS_FILE), exist_ok=True)

RESULTS_HEADER = [
    "name", "input_h", "input_w", "input_c", "output_c",
    "baseline_peak_mem", "peak_mem", "peak_mem_ratio", "peak_mem_saved", "board",
    "baseline_cycles", "cycles", "cycles_ratio",
    "baseline_duration_us", "duration_us", "duration_ratio",
    "baseline_stack_used", "stack_used",
    "baseline_checksum", "checksum", "checksums_agree",
]

# Written incrementally (one row per model, flushed immediately) rather than
# buffered until the end, so a board that stops responding partway through a
# sweep does not cost the models already measured.
results = []
with open(RESULTS_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(RESULTS_HEADER)

    for x in args.models:
        name = f"mcunet-in{x}"

        baseline_peak_mem = generate_model(x, cwd=args.baseline_comparison) \
                            if args.baseline_comparison else None
        peak_mem = generate_model(x)
        shape = read_model_shape()

        peak_ratio = (peak_mem / baseline_peak_mem) if baseline_peak_mem else None
        peak_saved = (baseline_peak_mem - peak_mem) if baseline_peak_mem else None
        if baseline_peak_mem:
            logger.info(
                f"{name}: peak activation memory {baseline_peak_mem} -> {peak_mem} bytes "
                f"({(1 - peak_ratio) * 100:+.1f}%)"
            )
        else:
            logger.info(f"{name}: peak activation memory {peak_mem} bytes")

        # Each tree keeps its own codegen/, so both still hold the model generated above.
        # Baseline flashed first.
        base_dev, dev = {}, {}
        if args.pico_board is not None:
            if args.baseline_comparison:
                base_dev = run_on_board(
                    x, baseline_peak_mem, args.pico_board, args.timeout, args.port,
                    cwd=args.baseline_comparison, label="baseline",
                ) or {}
            dev = run_on_board(
                x, peak_mem, args.pico_board, args.timeout, args.port, label="right-inplace"
            ) or {}
            if dev:
                logger.info(
                    f"{name}: {dev['duration_us']}us, {dev['cycles']} cycles, "
                    f"stack {dev['stack_used']}B, checksum {dev['checksum']}"
                )

        def _ratio(new, base):
            return (new / base) if (new is not None and base) else None

        base_check, check = base_dev.get("checksum"), dev.get("checksum")
        agree = (base_check == check) if None not in (base_check, check) else None
        if agree is False:
            logger.error(
                f"{name}: checksum mismatch, baseline {base_check} vs {check} -- "
                "the two builds did not compute the same result"
            )

        cycles_ratio = _ratio(dev.get("cycles"), base_dev.get("cycles"))
        duration_ratio = _ratio(dev.get("duration_us"), base_dev.get("duration_us"))

        row = {
            "name": name, **shape,
            "baseline_peak_mem": baseline_peak_mem, "peak_mem": peak_mem,
            "peak_mem_ratio": peak_ratio, "peak_mem_saved": peak_saved,
            "board": args.pico_board,
            "baseline_cycles": base_dev.get("cycles"), "cycles": dev.get("cycles"),
            "cycles_ratio": cycles_ratio,
            "baseline_duration_us": base_dev.get("duration_us"),
            "duration_us": dev.get("duration_us"), "duration_ratio": duration_ratio,
            "baseline_stack_used": base_dev.get("stack_used"),
            "stack_used": dev.get("stack_used"),
            "baseline_checksum": base_check, "checksum": check,
            "checksums_agree": agree,
        }
        results.append(row)

        writer.writerow([row[column] for column in RESULTS_HEADER])
        file.flush()

logger.info(f"wrote {len(results)}/{len(args.models)} model results to {RESULTS_FILE}")

# Summary statistics across the models swept. Only the series that were
# actually measured contribute; a codegen-only run summarises peak_mem alone.
SUMMARY_HEADER = ["metric", "mean", "min", "max", "median", "std"]

with open(SUMMARY_RESULTS_FILE, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(SUMMARY_HEADER)

    def _stats(values):
        arr = np.array(values, dtype=float)
        return arr.mean(), arr.min(), arr.max(), float(np.median(arr)), arr.std()

    for stat_name in ("baseline_peak_mem", "peak_mem", "peak_mem_saved", "peak_mem_ratio",
                      "baseline_cycles", "cycles", "cycles_ratio",
                      "baseline_duration_us", "duration_us", "duration_ratio",
                      "baseline_stack_used", "stack_used"):
        values = [r[stat_name] for r in results if r.get(stat_name) is not None]
        if values:
            mean, mn, mx, med, std = _stats(values)
            writer.writerow([stat_name, mean, mn, mx, med, std])

    # Cycles per byte of activation memory: how the runtime cost scales with
    # the footprint the schedule chose.
    ratio = [
        r["cycles"] / r["peak_mem"] for r in results
        if r.get("cycles") is not None and r.get("peak_mem")
    ]
    if ratio:
        mean, mn, mx, med, std = _stats(ratio)
        writer.writerow(["cycles/peak_mem", mean, mn, mx, med, std])

logger.info(f"wrote summary stats for {len(results)} models to {SUMMARY_RESULTS_FILE}")
