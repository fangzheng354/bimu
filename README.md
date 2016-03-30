# Bilingual Learning of Multi-sense Embeddings with Discrete Autoencoders

(c) Simon Šuster, 2016

This is the implementation of the embedding models described in:

 *Bilingual Learning of Multi-sense Embeddings with Discrete Autoencoders*. Simon Šuster, Ivan Titov and Gertjan van Noord. To appear at NAACL, 2016.

The individual similarity scores, presented as averages in the paper, are reported in [appendix](appendix/).

## Bilingual training of the multi-sense model
See `python3.4 examples/run_bimu.py --help` for the full list of options, and set the Theano flags as `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32`.


To train a multi-sense model bilingually on a toy parallel corpus of 10k sentences with default parameters:

```sh
python3.4 examples/run_bimu.py -model bimu -corpus_path_e data/toy_10k_en -corpus_path_f data/toy_10k_fr -corpus_path_a data/toy_10k_align -model_f_dir $OUTPUT_FR
``` 

This will output the embedding matrices, the vocabulary and the configuration file in the `output/bimu3_toy_10k_en_...` directory ($OUTPUT). Note that this presupposes that the second-language embeddings already exist in the folder $OUTPUT_FR. If not, train them by simply running:

```sh
python3.4 examples/run_mono.py -corpus_path data/toy_10k_fr -model sg
```

To obtain the nearest neighbors for selected polysemous words:

```sh
python3 eval/neighbors.py -input_dir $OUTPUT
```

To evaluate the embeddings on the SCWS dataset:

```
python3 eval/scws/embed.py -input_dir $OUTPUT -model senses -sim avg_exp
```

To train and test the POS tagger: 

```sh
python3 eval/nn/score.py -train_file wsjtrain -test_file wsjtest  
-tag_vocab_file data/tagvocab.json -embedding_file $OUTPUT/W_w.npy -vocab_file $OUTPUT/w_index.json -cembedding_file $OUTPUT/W_c.npy
```

Here, you will need the gold standard WSJ data available as `wsjtrain` and `wsjtest`. The index of POS tags is given as a json file, example can be found in `data/tagvocab.json`.

## Training the monolingual Skip-Gram and multi-sense models
See `python3.4 examples/run_mono.py --help` for the full list of options, and set the Theano flags as `THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32`.


To train the basic SkipGram with default options:

```sh
python3.4 examples/run_mono.py -corpus_path data/toy_10k_en -model sg
```

To train a multi-sense embedding model with 3 senses per word:

```sh
python3.4 examples/run_mono.py -corpus_path data/toy_10k_en -model senses -n_senses 3
```
