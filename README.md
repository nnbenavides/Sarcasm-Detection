# Sarcasm Detection

This repository contains code for 2 projects on sarcasm detection. The code at the main level of this repo was developed for the final project of Stanford's CS224N (Natural Language Processing with Deep Learning), which extended a previous course project and explored additional ways to extract contextual information using deeper models and attention. Code for the previous course project, which was a final project for Stanford's CS 224U (Natural Language Understanding), can be found in the old_project folder, and it explores the value of including context at the embedding level (comparing GloVe vs ELMo) and at the model-level (comparing a feed-forward NN with a bidirectional LSTM).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

What things you need to install the software and how to install them

```
1. Create a new Python3 virtual environment

Run python3 -m venv [your virtual environment name]
To activate your virtual environment, run  source [your virtual environment name]/bin/activate

2. Python >= 3.6

To check your python version, run python -V. If the Python version is less than 3.6, you will need to install a new version of python. You can follow this tutorial(https://realpython.com/installing-python/) if you need to upgrade your Python version.

3. numpy

If you have pip, you can simply run pip install numpy

4. NLTK

If you have pip, simply run pip install --user -U nltk

5. Pytorch 1.4.0

If you have pip, simply run pip install torch==1.4.0

6. scikit-learn

If you have pip, simply run pip install sklearn

7. allennlp (requires Python >= 3.6)

If you have pip, simply run pip install allennlp
```

### Getting Up and Running

1. Pull the data.
```
If you want to pull the full dataset (~128k examples), run sh dataset_scripts/pull_all_data.sh
If you want to pull a smaller version of the dataset (~6k examples, all political comments), run sh dataset_scripts/pull_small_data.sh
Either script should take a several minutes to run.
```

2. Generate ELMo embeddings for the data.

```
You'll need to run the elmo_embeddings.py script to generate ELMo embeddings for the data. If you pulled the full dataset, you can run the script as is by executing python elmo_embeddings.py. If you pulled the smaller dataset, you'll need to edit elmo_embeddings.py by changing the 'main' on line 17 to 'pol'. This may take several hours for the full dataset.
```

3. Start training models.
```
With the appropriate environment set up and a processed dataset, you're ready to start training models.
To run the training script, run python train.py -m [model_type] -e [error_file], specifying one of the existing model types and a file to output errors for error analysis.
Example: python train.py -m bilstm -e bilstm_errors.txt
```

## Built With

* [Pytorch](https://pytorch.org/) - Deep Learning Framework
* [AllenNLP](https://allennlp.org/) - NLP package used to generate ELMo embeddings

## Authors

* **Nicholas Benavides** - *Wrote most of the code for the projects*

See also the list of [contributors](https://github.com/nnbenavides/Sarcasm-Detection/graphs/contributors) who participated in this project.

## Acknowledgments

* Thanks to [kolchinski](https://github.com/kolchinski/reddit-sarc), [yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py), and [MLWhiz](https://gist.githubusercontent.com/MLWhiz/1ac0841f0333a97396d300b8f4c247c9/raw/aa352c54d00f801ea1579790652ff8ebb160b01b/pytorch_attention.py) for code used in various parts of these projects.
* Thanks to Chris Manning and the CS 224N teaching staff for their guidance and instruction Winter 2020.
* Thanks to Chris Potts and the CS 224U teaching staff for their guidance and instruction for Spring 2019.