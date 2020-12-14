#!/bin/bash

[ -z "$1" ] && echo "First argument should be directory with archive file." && exit 1
ARCHIVE_DIR="$1"

[ -z "$2" ] && echo "Second argument should be the vocabulary directory" && exit 1
VOCAB_DIR="$2"

rm -rf "$ARCHIVE_DIR/vocabulary" || mv "$ARCHIVE_DIR/vocabulary" "$ARCHIVE_DIR/artifact" || exit $?

mkdir "$ARCHIVE_DIR/archive" || exit $?

echo "Extracting the archive now ..."
tar xvzf "$ARCHIVE_DIR"/model.tar.gz --directory "$ARCHIVE_DIR/archive" || exit $?
rm -rf "$ARCHIVE_DIR"/archive/vocabulary || exit $?
cp -r "$VOCAB_DIR" "$ARCHIVE_DIR"/archive/vocabulary || exit $?
rm -f "$ARCHIVE_DIR"/model.tar.gz || exit $?
echo "Recreating the archive now ..."
cd "$ARCHIVE_DIR/archive"
tar cvzf model.tar.gz * || exit $?
mv model.tar.gz ..
