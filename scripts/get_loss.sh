#!/bin/bash

for i in `ls $1 `; do
  echo "$i"
  cat "$i" | awk -F ',' '
BEGIN { 
    id = 0
    s = 0
    p = 0
}
{
    if (id == 1)
    {
        s = $3
    }
    p = $3
    id = id + 1
}
END { print (s - p) / s }
'
done