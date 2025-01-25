#!/bin/bash

# Use sed to remove the first character and the last line in one go.
sed -e 's/.//' -e '$d' input.txt > tmp.txt

# Use tail to get all lines except the first and redirect to output.txt
tail -n +2 tmp.txt > output.txt

# Rename the temporary file to input.txt
mv tmp.txt input.txt
