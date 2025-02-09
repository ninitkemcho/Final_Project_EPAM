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

## MLE part instructions

## Data:

Run ` src/data_loader.py ` for generating and saving train, test data in csv format. 
It creates new folder ` data/raw ` and saves them locally.

## Training:

**Instructions for training the model in Docker:**

1. Build an image:

Make sure you are in your local repository directory and run this in dockerfile terminal

```
docker build -t training_image -f src/train/Dockerfile .
```

2. Run the container:

```bash
docker run -it training_image /bin/bash
```

To return to your host machine run:
```bash
exit
```

Create new directory for model
```bash
mkdir -p outputs/models
```

Save processed data and model locally
```bash
docker cp <container_id>:/app/data/processed ./data/
docker cp <container_id>:/app/outputs/models/model.pkl ./outputs/models/model.pkl
```

Do not forget to replace `<container_id>` with your running Docker container ID 


**Without Docker you can just run ` src/train/train.py `**

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

**Without Docker you can just run ` src/inference/run_inference.py `**

## DS part

### EDA

1. Explored unnecessary text breaks, common english words
2. Text cleaning included: removing unnecessary symbols, whitespaces and httml/URL tags

### Feature Engineering

1. After tokenization, stop words were removed and used lemmatization over stemming (almost same results, but lemmatization is considered to be less prone to errors)
2. Vectorization -> TF-IDF (because it showed better results, compared to n-grams, 2% higher accuracy)

### Modeling

1. Used Logistic Regression as baseline model, because its simple
2. Compared results of it to Random Forest and SVM (interesting with high-dimensional data)
3. Model Selection -> Logistic Regression (chose because of highest accuracy, I know my modeling part excludes cross-validation and better model could be found after that)


**Model results**
```
==================================================
Accuracy  : 0.8852
Precision : 0.8765
Recall    : 0.8968
F1-score  : 0.8865
AUC-ROC   : 0.9561
==================================================
```

You can see more comprehensive discussion in `notebooks/final_project.ipynb`

### Business Application

1. Recommendation systems for Netflix,etc to suggest positively rated movies and improve user experience
2. Theaters/movies can use these classifications to extend/cut the movie showing period
3. Imdb can implement this system for providing 'summarized' review and shorten the time for searching movie
4. Could be further modified for identifying spams, filtering real and fake reviews from each other. 