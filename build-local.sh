#!/bin/bash

SCRIPT_DIR=${0%/*}

cd $SCRIPT_DIR

BUILD_DIR=`pwd`/build

mkdir -p $BUILD_DIR/libfractal || exit;

cd $BUILD_DIR/libfractal || exit;

if [ ! -f Makefile ]; then
	../../libfractal/configure --prefix=$BUILD_DIR || exit;
fi

make install || exit;
