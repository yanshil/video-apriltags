#!/bin/bash

folderName="tag_output"

mkdir -p $folderName

for file in $(ls *.png)
do
	convert $file -scale 2000% $folderName/$file.png
done