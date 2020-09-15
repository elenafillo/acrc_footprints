FILE=$1


echo "Uncompressing [$FILE]"


LOC=/work/ef17148/acrc_footprints_samples/sample_checkpoints/$FILE.tar.gz
cp $LOC .
mkdir $FILE
tar -zxf $FILE.tar.gz -C $FILE
rm $FILE.tar.gz 
