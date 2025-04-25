#!/bin/bash
# cleanup
rm -rf build/
rm -rf dist/
rm -rf clemcore.egg-info/
# build
python -m build
# upload (insert api token manually)
twine upload dist/*
