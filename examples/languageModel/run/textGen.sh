#!/bin/bash

SCRIPT_DIR=${0%/*}

cd $SCRIPT_DIR

../bin/langModel -g 'The meaning of life is ' workspace
