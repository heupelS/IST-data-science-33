#!/bin/bash

# Stop container
if [ $( docker ps -a | grep anaconda_ds | wc -l ) -gt 0 ]; then
    echo 'Remove old container.'
    docker stop anaconda_ds
    docker rm -f anaconda_ds
    echo 'Remove successfull.'
fi

echo 'Create new container.'

docker run --rm -d \
    --name anaconda_ds \
    --volume="$PWD:/data-science" \
    anaconda_ds_image \
    sleep 999999999

echo 'New docker container successfully created.'