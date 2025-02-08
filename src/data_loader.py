import requests
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))
file_path_train = os.path.join(base_dir, 'train.csv')
file_path_test = os.path.join(base_dir, 'test.csv')

os.makedirs(base_dir, exist_ok=True)

def download_file(url, save_path):

    print(f'Downloading from {url}...')
    response = requests.get(url, stream=True)
    
    if response.ok:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f'Saved to {save_path}')
    else:
        raise Exception(f'Download failed: {response.status_code}')

if __name__ == '__main__':
    dataset_url_train = 'https://drive.google.com/uc?id=1kodw_qjeJfYP7urTNc7fwaLTkeYFiQQ8'
    dataset_url_test = 'https://drive.google.com/uc?id=1--F6dcmchqooYFc7UHtpmNn90pLuimy1'

    download_file(dataset_url_train, file_path_train)
    download_file(dataset_url_test, file_path_test)
    