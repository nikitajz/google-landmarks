#!/bin/bash
# default Ubuntu `parallel` is part of `moreutils` package and quite limited. I've used instead GNU `parallel`.
# sudo apt-get remove moreutils && sudo apt-get install parallel

# run inside the original directory
res=x224  # resolution, square e.g. 256x256
output_dir=../$res  # alternatively ../x256 if it should be placed in the parent dir

echo "Converting image to resolution $res, output_dir is $output_dir"

# TODO: mkdir should be done once per folder, not for each file
find . -name '*.jpg' | parallel --bar "mkdir -p $output_dir/{//}; convert -quality 98 -resize \"224x224\" {} $output_dir/{}"
