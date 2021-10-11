# Text2Music Emotion Embedding

Text-to-Music Retrieval using Pre-defined/Data-driven Emotion Embeddings
## Reference
Emotion Embedding Spaces for Matching Music to Stories, ISMIR 2021 [[arxiv](https://arxiv.org)]

-- Minz Won, Justin Salamon, Nicholas J. Bryan, Gautham J. Mysore, and Xavier Serra


## Requirements
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Data
- You need to collect audio files of AudioSet mood subset ([link](https://research.google.com/audioset/ontology/music_mood_1.html)).

- Read the audio files and store them into `.npy` format.

- Other relevant data including Alm's dataset, ISEAR dataset, emotion embeddings, pretrained Word2Vec, and data splits are available here ([link](https://github.com)).

- Unzip `ttm_data.tar.gz` and locate the extracted `data` folder under `text2music-emotion-embedding/`.


## Training
Check `scripts` folder to train models. 

For example:

`bash scripts/metric_learning.sh`

Do not forget to modify `YOUR_AUDIO_PATH`.

## Test
Here is an example for the test.

`python3 -u src/metric_learning/main.py --mode 'TEST' --dataset 'alm' --model_load_path 'data/pretrained/alm_cross.ckpt' --data_path 'YOUR_DATA_PATH'`

## Visualization


## Demo
We included a pretrained 3-branch metric learning model in the aforementioned link. Following examples are retrieval results of the model.

| Queried text      | Retrieved music |
| ----------- | ----------- |
| Hello world! <br/> This is an example text from ISEAR dataset.      | [YoutubeLink](https://youtube.com)       |


## License
```
Some License
```
