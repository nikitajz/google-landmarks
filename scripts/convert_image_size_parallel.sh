#!/bin/bash
# default Ubuntu `parallel` is part of `moreutils` package and quite limited. I've used instead GNU `parallel`.
# sudo apt-get remove moreutils && sudo apt-get install parallel

# run inside the original directory
res=x128  # resolution, square e.g. 256x256
output_dir=../$res  # alternatively ../x256 if it should be placed in the parent dir
cores=-6 # see "-P" option for parallel

echo "Creating subfolders"
find . -type d | parallel -P $cores --eta "mkdir -p $output_dir/{}"

echo "Converting images to resolution $res, output_dir is $output_dir"
find . -name '*.jpg' | parallel -P $cores --eta "convert -colorspace RGB -geometry $res {} $output_dir/{}"

echo "Copying the file 'train.csv'"
cp train.csv $output_dir/train.csv
echo "Done"