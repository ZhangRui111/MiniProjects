# Assemble Model by saver.restore()
Restore weights from pretrained model to build/assemble a new model.

For example, if you have trained a two-layer model before you train a three-layer model.
We can restore the weights from the two-layer model to initialize the first two layers
of the three-layer model.

## How to run
1. run `model_1.py` to save the initialized model for further use/assemble.
2. run `main.py`.