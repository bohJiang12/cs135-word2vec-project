# CS135 Final project: Word2Vec

**Author**: Bohan Jiang, Derrick Kim

## Description
- `data/`: raw dataset for training and evaluating embeddings
  - `SimLex-999`: [SimLex-999](https://fh295.github.io/simlex.html) dataset for evaluation
  - `wordsim353_sim_rel`: [WordSim-353](http://alfonseca.org/eng/research/wordsim353.html) dataset for evaluation
  - `questions-words.txt`: [Google's analogy test dataset](https://github.com/nicholas-leonard/word2vec/blob/master/questions-words.txt) for evaluating word embeddings
  - `train.txt`: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)
- `eval/`: contains modules for evaluation
  - `analogy.py`: run analogy test
  - `cos_sim.py`: compute consine similarity for word pairs of a certain dataset
  - `eval_utils.py`: utility modules for helping evaluation
  - `visualize.py`: run visualization for trained embeddings on a certain dataset
- `results/`: experiments results
- `src/`
  - `dataset.py`: script for processing and preparing datasets for training
  - `models.py`: model definition: CBOW and skip-gram
  - `utils.py`: utility functions for helping training the models
- `weights/`: contains trained models and their corresponding vocabulary
- `train.py`: main file for training models
- `evaluate.py`: main file for evaluation

## Usage
### Environment preparation
```
conda activate <project_env>
pip install -r requirements.txt
```

### Training
Given `config.yaml` file which specifies different aspects of training hyperparameters (e.g. window size, optimizer, learning rate, etc.), run the `train.py` as
```
python train.py --configFile config.yaml
```

### Evaluation
There are two options user must specify:
- `--model`: choose either CBOW or skip-gram
- `--dataset`: choose either SimLex-999 or WordSim-353

If we want to evaluate CBOW on dataset SimLex-999, run
```
python evaluate.py --model cbow --dataset simlex
```
Or if we are evaluating skip-gram on dataset WordSim-353, run
```
python evaluate.py --model sg --dataset wordsim
```
