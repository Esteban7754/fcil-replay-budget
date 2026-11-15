#!/bin/sh
# Image entry point.
#   docker run IMAGE                                   -> runs the unit tests
#   docker run IMAGE python plan_experimentos.py ...   -> runs the command as given
set -e

if [ "$#" -eq 0 ] || [ "$1" = "tests" ]; then
    exec python -m unittest discover -p "test_*.py" -v
fi
exec "$@"
