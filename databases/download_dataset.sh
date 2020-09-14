FILE=$1

if [[ $FILE != "met_to_fp_noise" && $FILE != "met_to_fp_noiseless" && $FILE != "press_to_fp_noise" && $FILE != "press_to_fp_noiseless" && $FILE != "press_to_fp_noiseless_small"  ]]; then
  echo "The input is not an available dataset. Please check the available datasets and try again"
  exit 1
fi


echo "Uncompressing [$FILE]"



tar -zxf ./sample_datasets/$FILE.tar.gz 
