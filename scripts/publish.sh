#!/bin/bash

rm -rf dist
python -m build


# This will ask for a password, which is the api token
python -m twine upload -u __token__ dist/*

