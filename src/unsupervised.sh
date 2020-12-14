#!/bin/bash

[ -z "$1" ] && echo "First argument should be serialization directory." && exit 1
SER_DIR="$1"

[ -z "$2" ] && echo "Second argument should be the config file for iteration 0." && exit 1
LM_CONFIG="$2"

[ -z "$3" ] && echo "Third argument should be the config file for iterations >0." && exit 1
ITER_CONFIG="$3"

[ -z "$4" ] && echo "Fourth argument should be the unlabeled corpus." && exit 1
UNSUP_DATA="$4"

[ -z "$5" ] && echo "Fifth argument should be the config file for backtranslation." && exit 1
BT_CONFIG="$5"

[ -z "$6" ] && echo "Sixth argument should be the number of iterations." && exit 1
NUM_ITERATION="$6"

[ -z "$7" ] && echo "Seventh argument should be the vocabulary directory." && exit 1
VOCAB_DIR="$7"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

while true; do
    echo "Serialization dir: $SER_DIR"
    echo "Config file for iteration 0: $LM_CONFIG"
    echo `python3 "$DIR/show_data_path.py" "$LM_CONFIG"`
    echo "Config file for iteration 1: $ITER_CONFIG"
    echo `python3 "$DIR/show_data_path.py" "$ITER_CONFIG"`
    echo "Unlabeled data to use: $UNSUP_DATA"
    echo "Config file for backtranslation: $BT_CONFIG"
    echo "Vocabulary directory: $VOCAB_DIR"
    read -p "Is this ok [y/n] ? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit 0;;
        * ) echo "Please answer y or n.";;
    esac
done

echo "ACTION REPORT: pretrain language models"
allennlp train "$LM_CONFIG" -s "$SER_DIR/iter0" --include-package src.features.copynet_shared_decoder --include-package src.models.copynet_shared_decoder || exit $?

echo "ACTION REPORT: start training the iterations"
. "$DIR/unsup-iterations.sh" "$SER_DIR" "$ITER_CONFIG" "$UNSUP_DATA" "1" "$NUM_ITERATION" "$BT_CONFIG" "$VOCAB_DIR" "NOPROMPT"
