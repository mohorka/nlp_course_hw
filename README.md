# Tasks solutions for NLP course
## Prerequisites
Make sure you have `poetry` installed. All solutions below ran with Python 3.11.3
1. Clone repo
2. Create virtual enviroment inside cloned repo (setup docker container or simply run `python3 -m venv .venv && source .venv/bin/activate`)
3. Inside virtual enviroment: `poetry install`.
## Task1: extract person data with finite state automata
#### How to run?
Run `parser -i path_to_input_file.txt -o output.txt`.
To protect from corrupted results, it's forbidden to write result from many script's runs to the same file.