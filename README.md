# Text2Music Emotion Embedding

Text-to-Music Retrieval using Pre-defined/Data-driven Emotion Embeddings
## Reference
Emotion Embedding Spaces for Matching Music to Stories, ISMIR 2021 [[paper](https://arxiv.org/abs/2111.13468)]

-- Minz Won, Justin Salamon, Nicholas J. Bryan, Gautham J. Mysore, and Xavier Serra

```
@inproceedings{won2021emotion,
  title={Emotion embedding spaces for matching music to stories},
  author={Won, Minz. and Salamon, Justin. and Bryan, Nicholas J. and Mysore, Gautham J. and Serra, Xavier.},
  booktitle={ISMIR},
  year={2021}
}
```


## Requirements
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Data
- You need to collect audio files of AudioSet mood subset ([link](https://research.google.com/audioset/ontology/music_mood_1.html)).

- Read the audio files and store them into `.npy` format.

- Other relevant data including Alm's dataset ([original link](http://people.rc.rit.edu/~coagla/affectdata/index.html)), ISEAR dataset ([original link](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/)), emotion embeddings, pretrained Word2Vec, and data splits are all available here ([link](https://www.dropbox.com/sh/qcc30tjy256ursl/AABr0wOyzgxoASP40H1EmWN8a)).

- Unzip `ttm_data.tar.gz` and locate the extracted `data` folder under `text2music-emotion-embedding/`.


## Training
Here is an example for training a metric learning model.

```
python3 src/metric_learning/main.py \
        --dataset 'isear' \
        --num_branches 3 \
        --data_path YOUR_DATA_PATH_TO_AUDIOSET
```

Fore more examples, check bash files under `scripts` folder. 

## Test
Here is an example for the test.

```
python3 src/metric_learning/main.py \
        --mode 'TEST' \
        --dataset 'alm' \
        --model_load_path 'data/pretrained/alm_cross.ckpt' \
        --data_path 'YOUR_DATA_PATH_TO_AUDIOSET'
```
Pretrained three-branch metric learning models (`alm_cross.ckpt` and `isear_cross.ckpt`) are included in `ttm_data.tar.gz`. This code is reproducible by locating the unzipped `data` folder under `text2music-emotion-embedding/`.

## Visualization
Embedding distribution of each model can be projected onto 2-dimensional space. We used uniform manifold approximation and projection (UMAP) to visualize the distribution. UMAP is known to preserve more of global structure compared to t-SNE.


<p align = "center">
<img src = "https://imgur.com/wNlXG6I.png">
</p>



## Demo
Please try some examples done by the three-branch metric learning model [[Soundcloud](https://soundcloud.com/minz-won/sets/emotion-embedding-spaces-for-matching-music-to-stories-demo?si=7249b1881611425da41c63ac0d123305)].



## License
```
Text2Music Emotion Embedding

MIT License
Copyright (c) 2021 Music Technology Group, Universitat Pompeu Fabra. 
Code developed by Minz Won.

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```
