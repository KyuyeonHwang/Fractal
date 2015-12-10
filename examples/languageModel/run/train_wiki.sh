#!/bin/bash

SCRIPT_DIR=${0%/*}

cd $SCRIPT_DIR

../bin/langModel -t ../data/wiki_small workspace
