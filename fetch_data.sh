#!/bin/sh

pushd `dirname $0` >/dev/null
BASEDIR=`pwd`
popd >/dev/null

>&2 echo "Downloading parameterization data to $BASEDIR..."

rsync -avz --cvs-exclude data.icecube.wisc.edu:/data/user/jvansanten/projects/2015/gen2_benchmark/data/ $BASEDIR/resources/data

>&2 echo "done"
