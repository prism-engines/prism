#!/bin/bash
# PRISM Ingest - Quick file processing with Claude
#
# Usage:
#     ./scripts/ingest.sh mydata.csv
#     ./scripts/ingest.sh /path/to/sensors.parquet
#
# This will launch Claude CLI and analyze the file using prism.intake

if [ -z "$1" ]; then
    echo "Usage: $0 <data_file>"
    echo ""
    echo "Supported formats: .csv, .tsv, .parquet"
    echo ""
    echo "Example:"
    echo "  $0 pump_data.csv"
    echo "  $0 ~/Downloads/sensors.parquet"
    exit 1
fi

FILE="$1"

# Resolve to absolute path
if [[ ! "$FILE" = /* ]]; then
    FILE="$(pwd)/$FILE"
fi

if [ ! -f "$FILE" ]; then
    echo "Error: File not found: $FILE"
    exit 1
fi

# Check extension
EXT="${FILE##*.}"
case "$EXT" in
    csv|tsv|parquet|pq)
        ;;
    *)
        echo "Warning: Unexpected extension .$EXT (expected csv, tsv, or parquet)"
        ;;
esac

echo "Processing: $FILE"
echo ""

# Run Claude with intake prompt
exec claude "Process this data file with PRISM intake:

File: $FILE

1. Run intake analysis: from prism.intake import intake; result = intake('$FILE'); print(result.summary())
2. Show detected signals, units, entities
3. Run sanity checks: from prism.sanity import check_dataframe
4. Recommend domain config based on detected signals/units
5. Report what physics computations would be possible given the data

Start by reading and analyzing the file."
