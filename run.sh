#!bash

set -euo pipefail

stage=0
data=data

# Set your authentication info.
db_user=
db_passwd=

. parse_options.sh

if [ $stage -le 0 ]; then
    echo "=== stage 0: download ==="

    if [ -z "$db_user" ] || [ -z "db_passwd" ]; then
	echo "ERROR: these flags are required: $ ./run.sh --db_user --db_passwd"
	echo "Register them at http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database"
	exit 1
    fi
    
    mkdir -p "$data"
    cd "$data"
    for file in ascii-all.tar.gz original-xml-all.tar.gz lineStrokes-all.tar.gz; do
	if [ -e "$file" ]; then
	    echo "$file already exists."
	    continue
	fi
	wget --http-user="$db_user" --http-password="$db_passwd" \
	     "http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/$file"
	tar xvf "$file"
    done
    cd ..
fi

if [ $stage -le 1 ]; then
    echo "=== stage 1: trainig ==="

    python3 ./train.py -v=1 --root=./data
fi

if [ $stage -le 2 ]; then
    echo "=== stage 2: inference ==="

    echo "Not implemented."
fi
