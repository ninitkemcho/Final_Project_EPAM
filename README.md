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


## Data:

Run ` src/data_loader.py ` for generating and saving train, test data in csv format. 

## Training:

Training is done in `train.py`

**Instructions for training the model in Docker:**

1. Build an image:

```bash
docker build --no-cache -f <path_to_your_local_training_dockerfile_directory> -t training_image <path_to_your_local_repository_directory>

```
My dockerfile has .dockerfile expansion, so be sure to include it in the path

Do not forget to replace <path_to_your_local_training_dockerfile_directory> and <path_to_your_local_repository_directory> with your local path


2. Run the container:

Run this code in cmd, because in my case it did not work in git bash.

```bash
docker run -it training_image /bin/bash
```

```bash
docker cp <container_id>:/app/model.pkl ./model.pkl
```

Replace `<container_id>` with your running Docker container ID 

**Without Docker using just Python:**

```bash
python3 training/train.py
```
**Not recommended, directories does not work**

## Inference


**Instructions for running the inference in Docker:**

1. Build the inference Docker image:

```bash
docker build --no-cache -f <path_to_your_local_inference_dockerfile_directory> --build-arg model_name=model.pkl -t inference_image <path_to_your_local_repository_directory>
```

2. Run the inference Docker container:

```bash
docker run -it inference_image /bin/bash  
```

After that ensure that there is `iris_predictions.csv` in the `results` directory in the inference container.


**Without Docker using just Python:**

```bash
python inference/run.py
```
**Not recommended, directories does not work**

## Conclusions

There was no need for initial data processing, because of simple data structure. Also, deep learning algorithm handled extremely well this small dataset and reached unrealistic accuracy metric. Not much to infer from so small data.
