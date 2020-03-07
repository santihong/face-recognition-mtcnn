## How to use
1. Download dataset [5-celebrity-faces-dataset](https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset) to `res` directory, or you can collect your own dataset.
   Dataset directory structure:
   - dataset-name
      - train
        - person1
          - xxx.jpg
          - xxx.jpg
          ...
        - person2
        ...
        - personM
      - val
        - person1
        - person2
        ...
        - personM
2. Run `python preprocess.py` to retrieve faces
3. Run `python embeddings.py` to generate embedding features
4. Run `main.py` to have fun