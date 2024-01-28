#!/bin/zsh
for file in *.xml
do
  iconv -f windows-1251 -t UTF-8 "$file" > "../decoded/${file%.xml}-utf8.xml"
done
