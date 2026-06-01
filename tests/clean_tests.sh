find . -type f -iname '*.pdf' -exec rm -v {} \;
find . -type f -iname '*.png' -exec rm -v {} \;
find . -type f -iname '*.csv' -exec rm -v {} \;
find . -type d -name 'outputs' -exec rm -rfv {} \;
