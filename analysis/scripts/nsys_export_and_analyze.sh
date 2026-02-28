#!/bin/bash
# nsys_export_and_analyze.sh -- Export nsys-rep to SQLite and run analysis queries
#
# Run on ACES where nsys is available:
#   module load nsight-systems
#   bash analysis/nsys_export_and_analyze.sh
#
# Analyzes: H2D bandwidth, NCCL timing, overlap detection, bandwidth during/without NCCL, GPU idle gaps
# Output: text reports + CSV results in results/profiles/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROFILES_DIR="$PROJECT_DIR/results/profiles"
OUTPUT_DIR="$PROFILES_DIR/analysis"

NEW_NSYS="$PROFILES_DIR/new_offload_nsys_1459999.nsys-rep"
OLD_NSYS="$PROFILES_DIR/old_offload_nsys_1460000.nsys-rep"

mkdir -p "$OUTPUT_DIR"

# Check nsys is available
if ! command -v nsys &>/dev/null; then
	echo "ERROR: nsys not found. Load it with: module load nsight-systems"
	exit 1
fi

echo "================================================================"
echo "  nsys Profile Export and Analysis"
echo "  New: $(basename "$NEW_NSYS")"
echo "  Old: $(basename "$OLD_NSYS")"
echo "================================================================"

# ===========================================================================
# Export nsys-rep to SQLite
# ===========================================================================
export_to_sqlite() {
	local nsys_file="$1"
	local sqlite_file="${nsys_file%.nsys-rep}.sqlite"

	if [ -f "$sqlite_file" ]; then
		echo "  SQLite already exists: $(basename "$sqlite_file")" >&2
	else
		echo "  Exporting $(basename "$nsys_file") -> SQLite..." >&2
		nsys export --type=sqlite --output="$sqlite_file" "$nsys_file" >&2
		echo "  Done: $(du -h "$sqlite_file" | cut -f1)" >&2
	fi
	echo "$sqlite_file"
}

echo ""
echo "--- Exporting to SQLite ---"
NEW_DB=$(export_to_sqlite "$NEW_NSYS")
OLD_DB=$(export_to_sqlite "$OLD_NSYS")

# ===========================================================================
# Run all queries via Python (sqlite3 CLI not available on ACES)
# ===========================================================================
echo ""
echo "--- Running SQL queries via Python sqlite3 module ---"
python3 "$SCRIPT_DIR/nsys_query.py" "$NEW_DB" "$OLD_DB" "$OUTPUT_DIR"
