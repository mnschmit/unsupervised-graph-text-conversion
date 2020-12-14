#!/bin/bash

[ -z "$1" ] && echo "1st argument should be text directory." && exit 1
TEXT_DIR="$1"

[ -z "$2" ] && echo "2nd argument should be the eval command." && exit 1
EVAL_SCRIPT="$2"

IFS=' ' read -ra COMMAND <<< "$EVAL_SCRIPT"

for ITER_DIR in "$TEXT_DIR"/iter* ; do
    ITER=`basename "$ITER_DIR"`
    echo "$ITER"
    "${COMMAND[@]}" "$ITER_DIR"/hypo.txt "$ITER_DIR"/ref.txt > /dev/null || exit $?
done
