# aspect-extraction
Model tried for the task of Aspect Extraction

## How to run examples
### LSTM CRF with POS Tags as features
Put `lstm_crf_pos_run.py`, `word2id.pickle` and `best_model_lstm_crf_pos.pt` files in same directory.
Run `lstm_crf_pos_run.py` with sentence within quotes as a command line argument. (Python 2.7)

Eg : ``python lstm_crf_pos_run.py "I like itallian pizza"``
