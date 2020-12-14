#!/bin/bash

[ -z "$1" ] && echo "First argument should be serialization directory." && exit 1
SER_DIR="$1"

[ -z "$2" ] && echo "Second argument should be the config file for iterations >0." && exit 1
ITER_CONFIG="$2"

[ -z "$3" ] && echo "Third argument should be the unlabeled corpus." && exit 1
UNSUP_DATA="$3"

[ -z "$4" ] && echo "Fourth argument should be the number of the first iteration." && exit 1
START_ITERATION="$4"

[ -z "$5" ] && echo "Fifth argument should be the number of iterations to be done." && exit 1
NUM_ITERATION="$5"

[ -z "$6" ] && echo "Sixth argument should be the config file for backtranslation." && exit 1
BT_CONFIG="$6"

[ -z "$7" ] && echo "Seventh argument should be the vocabulary directory." && exit 1
VOCAB_DIR="$7"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
let END_ITERATION=$START_ITERATION+$NUM_ITERATION-1

if [ -z "$8" ]; then
    while true; do
	echo "Serialization dir: $SER_DIR"
	echo "Config file for iterations: $ITER_CONFIG"
	python3 "$DIR/show_data_path.py" "$ITER_CONFIG"
	echo "Unlabeled data to use: $UNSUP_DATA"
	echo "Next iteration: $START_ITERATION"
	echo "Final iteration: $END_ITERATION"
	echo "Config file for backtranslation: $BT_CONFIG"
	echo "Vocabulary directory: $VOCAB_DIR"
	read -p "Is this ok [y/n] ? " yn
	case $yn in
            [Yy]* ) break;;
            [Nn]* ) exit 0;;
            * ) echo "Please answer y or n.";;
	esac
    done
fi


CUR_CONFIG="$ITER_CONFIG"
for i in $(seq $START_ITERATION $END_ITERATION)
do
    CUR_ITER=$i
    let PREV_ITER=$CUR_ITER-1

    echo "ACTION REPORT: backtranslate with model from iter${PREV_ITER}"
    python3 -m src.data.backtranslate "$BT_CONFIG" "${SER_DIR}/iter${PREV_ITER}/best.th" "$UNSUP_DATA" "$SER_DIR/train-bt${CUR_ITER}.tsv" || exit $?

    echo "ACTION REPORT: fine-tune iter${CUR_ITER} model on the backtranslated data"
    allennlp fine-tune -m "$SER_DIR/iter${PREV_ITER}/model.tar.gz" -c "$CUR_CONFIG" -s "$SER_DIR/iter${CUR_ITER}" --include-package src.features.copynet_shared_decoder --include-package src.models.copynet_shared_decoder || exit $?

    echo "Reparing newly created archive - just in case"
    src/repare_archive.sh "$SER_DIR/iter${CUR_ITER}" "$VOCAB_DIR"
    
    CUR_CONFIG=`python3 $DIR/generate_next_config.py $CUR_CONFIG`
    echo "ACTION REPORT: Created new config in ${CUR_CONFIG}"
done
