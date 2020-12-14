#!/bin/bash

[ -z "$1" ] && echo "1st argument should be serialization directory." && exit 1
SER_DIR="$1"

[ -z "$2" ] && echo "2nd argument should be the config file for backtranslation." && exit 1
CONFIG="$2"

[ -z "$3" ] && echo "3rd argument should be the dataset." && exit 1
DATASET="$3"

[ -z "$4" ] && echo "4th argument should be the output directory." && exit 1
OUTPUT_DIR="$4"

for ITER_DIR in "$SER_DIR"/iter* ; do
    ITER=`basename "$ITER_DIR"`
    echo "Generating with model from $ITER"
    mkdir -p "$OUTPUT_DIR"/"$ITER"
    python3 -m src.analysis.predict "$CONFIG" "$ITER_DIR"/best.th "$DATASET" "$OUTPUT_DIR"/"$ITER"/hypo.txt "$OUTPUT_DIR"/"$ITER"/ref.txt 64 || exit $?
done
