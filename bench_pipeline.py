#!/usr/bin/env python3
"""
Pipeline Benchmark — per-stage timing on FD_004.

Usage:
    ./venv/bin/python bench_pipeline.py ~/domains/cmapss/FD_004_bench/train
"""

import sys
import os
import time
import importlib
import traceback
from pathlib import Path

# Force unbuffered stdout
os.environ['PYTHONUNBUFFERED'] = '1'

import yaml

from manifold.run import ALL_STAGES, _out, _dispatch, _write_readmes
from manifold.io.reader import STAGE_DIRS


def main():
    data_path = Path(sys.argv[1])
    manifest_path = data_path / 'manifest.yaml'

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    manifest['_manifest_path'] = str(manifest_path)
    manifest['_data_dir'] = str(data_path)

    output_dir = data_path / manifest.get('paths', {}).get('output_dir', 'output')
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in sorted(set(STAGE_DIRS.values())):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    obs_path = str(data_path / manifest['paths']['observations'])

    # Check Rust status
    try:
        from manifold_rs import individual
        rust_status = "LOADED"
    except ImportError:
        rust_status = "NOT AVAILABLE"

    print("=" * 70)
    print("MANIFOLD PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Data:     {data_path}")
    print(f"Output:   {output_dir}")
    print(f"Stages:   {len(ALL_STAGES)}")
    print(f"Rust:     {rust_status}")
    print()

    results = []
    total_start = time.perf_counter()

    for module_path, stage_id in ALL_STAGES:
        stage_name = module_path.split('.')[-1]
        label = f"{stage_id} ({stage_name})"

        t0 = time.perf_counter()
        status = "OK"
        try:
            module = importlib.import_module(f'manifold.stages.{module_path}')
            if not hasattr(module, 'run'):
                status = "NO run()"
            else:
                _dispatch(module, module_path, stage_id, obs_path,
                          output_dir, manifest, verbose=False, data_path_str=str(data_path))
        except Exception as e:
            status = f"ERROR: {e}"
            traceback.print_exc()

        elapsed = time.perf_counter() - t0
        results.append((label, elapsed, status))
        print(f"  {label:<40s}  {elapsed:7.1f}s  {status}", flush=True)

    total_elapsed = time.perf_counter() - total_start
    _write_readmes(output_dir)
    print()
    print("-" * 70)
    print(f"  {'TOTAL':<40s}  {total_elapsed:7.1f}s")
    print("=" * 70)

    # Count output files
    total_files = 0
    for subdir in sorted(set(STAGE_DIRS.values())):
        files = list((output_dir / subdir).glob('*.parquet'))
        total_files += len(files)
        if files:
            print(f"  {subdir}/ ({len(files)} files)")
    print(f"\n  Total: {total_files} parquet files")

    # Summary table for copy-paste
    print("\n\nPer-Stage Timing Summary:")
    print(f"{'Stage':<40s}  {'Time':>8s}  {'Status'}")
    print("-" * 65)
    for label, elapsed, status in results:
        print(f"{label:<40s}  {elapsed:7.1f}s  {status}")
    print("-" * 65)
    print(f"{'TOTAL':<40s}  {total_elapsed:7.1f}s")


if __name__ == '__main__':
    main()
