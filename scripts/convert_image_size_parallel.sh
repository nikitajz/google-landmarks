#!/bin/bash
# default Ubuntu `parallel` is part of `moreutils` package and quite limited. I've used instead GNU `parallel`.
# sudo apt-get remove moreutils && sudo apt-get install parallel

res=x64  # resolution, square e.g. 256x256
output_dir=$res  # alternatively ../x256 if it should be placed in the parent dir

echo "Converting image to resolution $res"

# TODO: echo and mkdir should be done once per folder, not for each file
# TODO: rewrite to process subfolder (e.g. orig) to skip folders with other converted images
find . -maxdepth 6 -type f -name '*.jpg' | parallel "echo {}; mkdir -p $output_dir/{//}; convert -geometry $res {} $output_dir/{}"
