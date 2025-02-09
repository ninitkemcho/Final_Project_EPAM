# Final_Project_EPAM

## Project structure:

```
Final_Project_EPAM
├── data
│   ├── raw/
│   ├── processed/ # Includes clean, splitted and vectorized data
├── notebooks/
├── src
│   ├── train
│   │   ├── train.py
│   │   └── Dockerfile
│   ├── inference
│   │   ├── run_inference.py
│   │   └── Dockerfile
│   └── data_loader.py
├── outputs # Respository does not include, will be generated after running codes
│   ├── models/ 
│   │   └── model.pkl
│   ├── predictions/
│   │   ├── predictions.csv
│   │   └── metrics.txt
│   └── figures/ #Includes confusion matrix
├── README.md
└── requirements.txt
```

## Instructions

## Data:

Run ` src/data_loader.py ` for generating and saving train, test data in csv format. 
It creates new folder ` data/raw ` and saves them locally.

## Training:

Training is done in ` src/train/train.py`

**Instructions for training the model in Docker:**

1. Build an image:

Make sure you are in your local repository directory and run this in dockerfile terminal (I didn't know that last time)

```
docker build -t training_image -f src/train/Dockerfile .
```
'C:/Users/99559/OneDrive - iset.ge/Desktop/Extra Activities/Epam/Data Science/Final_Project_EPAM'

2. Run the container:

```bash
docker run -it training_image /bin/bash
```

To return to your host machine run:
```bash
exit
```

Create new directory for mode;
```bash
mkdir -p outputs/models
```

Save processed data and model locally
```bash
docker cp <container_id>:/app/data/processed ./data/
docker cp <container_id>:/app/outputs/models/model.pkl ./outputs/models/model.pkl
```

Do not forget to replace `<container_id>` with your running Docker container ID 


**Without Docker you can just run ` src/train/train.py ` **

## Inference


**Instructions for running the inference in Docker:**

1. Build the inference Docker image:

```bash
docker build -t inference_image -f src/inference/Dockerfile .
```

2. Run the inference Docker container:

```bash
docker run -it inference_image /bin/bash  
```

To return to your host machine run:
```bash
exit
```

Create new directories for results
```bash
mkdir -p outputs/figures
mkdir -p outputs/predictions
```

Saves results locally: metrics,cm,predictions
```bash
docker cp <container_id>:/app/outputs/predictions/predictions.csv ./outputs/predictions/predictions.csv
docker cp <container_id>:/app/outputs/predictions/metrics.txt ./outputs/predictions/metrics.txt
docker cp <container_id>:/app/outputs/figures/confusion_matrix.png ./outputs/figures/confusion_matrix.png
```

**Without Docker you can just run ` src/inference/run_inference.py ` **

## DS part

1. Text cleaning included: removing unnecessary symbols, whitespaces and httml/URL tags
2. After tokenization, stop words were removed and used lemmatization over stemming (almost same results, but lemmatization is considered to be less prone to errors)
3. Vectorization -> TF-IDF (because it showed better results, compared to n-grams)
4. Model Selection -> Logistic Regression 