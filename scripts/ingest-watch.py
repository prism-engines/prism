#!/usr/bin/env python3
"""
PRISM Ingest Watcher

Monitors a folder for new files and auto-triggers Claude CLI to process them.

Usage:
    # Watch default inbox folder
    python scripts/ingest-watch.py

    # Watch custom folder
    python scripts/ingest-watch.py /path/to/watch

    # One-shot mode (process existing files, don't watch)
    python scripts/ingest-watch.py --once

The watcher will:
1. Detect new CSV, TSV, or Parquet files
2. Call Claude CLI with the intake prompt
3. Move processed files to inbox/done/
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# Default inbox location
DEFAULT_INBOX = Path.home() / "prism-inbox"

# File extensions to process
VALID_EXTENSIONS = {'.csv', '.tsv', '.parquet', '.pq'}

# Prompt template for Claude
INTAKE_PROMPT = '''Process this data file with PRISM intake:

File: {file_path}

1. Run intake analysis using prism.intake
2. Show the IntakeResult summary
3. Recommend appropriate domain config based on detected signals/units
4. If units are detected, show what physics computations would be possible

Do NOT modify any files - just analyze and report.
'''


def get_files_to_process(inbox: Path) -> list[Path]:
    """Get list of files ready for processing"""
    files = []
    for ext in VALID_EXTENSIONS:
        files.extend(inbox.glob(f"*{ext}"))
    # Sort by modification time (oldest first)
    return sorted(files, key=lambda f: f.stat().st_mtime)


def process_file(file_path: Path, done_dir: Path, dry_run: bool = False) -> bool:
    """Process a single file with Claude CLI"""
    print(f"\n{'='*60}")
    print(f"Processing: {file_path.name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    prompt = INTAKE_PROMPT.format(file_path=file_path)

    if dry_run:
        print(f"[DRY RUN] Would run: claude --print '{prompt[:50]}...'")
        return True

    try:
        # Run Claude CLI
        result = subprocess.run(
            ["claude", "--print", prompt],
            cwd=str(file_path.parent.parent),  # Run from prism root if possible
            capture_output=False,  # Stream output to terminal
            text=True,
        )

        if result.returncode == 0:
            # Move to done folder
            done_path = done_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_path.name}"
            file_path.rename(done_path)
            print(f"\n[OK] Moved to: {done_path}")
            return True
        else:
            print(f"\n[ERROR] Claude exited with code {result.returncode}")
            return False

    except FileNotFoundError:
        print("[ERROR] Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def watch_folder(inbox: Path, done_dir: Path, poll_interval: float = 2.0):
    """Watch folder for new files"""
    print(f"Watching: {inbox}")
    print(f"Done dir: {done_dir}")
    print(f"Extensions: {', '.join(VALID_EXTENSIONS)}")
    print(f"Poll interval: {poll_interval}s")
    print("\nDrop files into the inbox folder to process them...")
    print("Press Ctrl+C to stop\n")

    seen_files = set(get_files_to_process(inbox))

    try:
        while True:
            current_files = set(get_files_to_process(inbox))
            new_files = current_files - seen_files

            for file_path in sorted(new_files, key=lambda f: f.stat().st_mtime):
                # Wait a moment for file to finish writing
                time.sleep(0.5)
                if file_path.exists():  # Still there after delay
                    process_file(file_path, done_dir)

            seen_files = set(get_files_to_process(inbox))
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n\nStopped watching.")


def main():
    parser = argparse.ArgumentParser(
        description="Watch folder for new data files and process with Claude"
    )
    parser.add_argument(
        "inbox",
        nargs="?",
        type=Path,
        default=DEFAULT_INBOX,
        help=f"Folder to watch (default: {DEFAULT_INBOX})"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process existing files once, don't watch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running Claude"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Poll interval in seconds (default: 2.0)"
    )

    args = parser.parse_args()
    inbox = args.inbox.expanduser().resolve()
    done_dir = inbox / "done"

    # Create directories
    inbox.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(exist_ok=True)

    if args.once:
        # One-shot mode
        files = get_files_to_process(inbox)
        if not files:
            print(f"No files to process in {inbox}")
            return

        print(f"Processing {len(files)} file(s)...")
        for f in files:
            process_file(f, done_dir, dry_run=args.dry_run)
    else:
        # Watch mode
        watch_folder(inbox, done_dir, poll_interval=args.interval)


if __name__ == "__main__":
    main()
