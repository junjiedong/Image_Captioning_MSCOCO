#!/usr/bin/env bash

# Remove unused files
cd coco/annotations && rm -f instances* && rm -f person*

# Re-organize the image and annotation files
# Moving the files around can take a little time, just be patient
cd .. && mv train2014 images/train && mv val2014 images/val

# Build the Python API
# There are some warnings when building the API, but it doesn't seem to cause problems
cd PythonAPI && make && cd ..