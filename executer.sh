#!/bin/bash  
while [ true ]
do
    go build
    buildok=$?
    if [ $buildok == 0 ]
    then
        echo "BUILDOK = $buildok"
        ./img-classificator
    else
        echo "ERROR COMPILING! ==============================="
    fi
    sleep 3
done