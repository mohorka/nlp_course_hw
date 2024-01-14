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
## Task2: text classification with word2vec/doc2vec as feature extractors.
#### How to run?
Here is 2 scripts: one for combination of mean of word2vec vectors per document + logistic regression for classification:

`dummy_classify -i path_to_input_file.txt --trained_w2v path_to_trained_w2v`

You can use your pretrained model or start fresh training with change arguments:
`dummy_classify -i path_to_input_file.txt --path_to_save_w2v path_to_fresh_w2v`

Second script uses as features excrator doc2vec model. Again, you can use your pretrained model, or choose fresh run:

`docs_classify -i path_to_input_file.txt --trained_d2v path_to_trained_d2v`

`docs_classify -i path_to_input_file.txt --path_to_save_d2v path_to_fresh_d2v`

Both scripts perform around 0.81 accuracy on given `news.txt` data.

## Task3: text generation with casual and masked LM.
To solve this problem, the following strategy was chosen: 
1. Process given sentence with LLM (used `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with instruction prompt (see notebook for details).
Actually, it is recommended to use `torch.bfloat16` model, but unfortunatelly, this is not supported for ARM, so `torch.float16` is used.
1. Naively post-process LLM output for receive generated paraphrased sentences.
2. Randomly masked a word in each paraphrased sentence.
3. Feed masked sentence to mask-LM (used `distilroberta-base`), take first two answers for each masked sentence.

Notebook with solution is in `task3` directory.