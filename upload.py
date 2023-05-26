import argparse
import os
import shutil
import yaml
from zipfile import ZipFile 
from utils import upload_dataverse_data

import zipfile

def double_zip_folder(folder_path, output_path):
    # Create a temporary zip file
    shutil.make_archive(output_path, 'zip', folder_path)

    # Zip the temporary zip file
    zipzip_path = output_path + '.zip.zip'
    with ZipFile(zipzip_path, 'w') as f:
        f.write(output_path + '.zip')

    # Remove the temporary zip file
    os.remove(output_path + '.zip')

    return zipzip_path

def main(args):
    # load metadata
    with open(f"{args.train_output_path}/metadata.yaml", 'r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
    
    # create folder
    if not os.path.exists(metadata['name']):
        os.mkdir(metadata['name'])
        #metadata.yaml
        shutil.copy(f"{args.train_output_path}/metadata.yaml", 
                    f"{metadata['name']}/metadata.yaml")
        #synthetic_data.csv
        shutil.copy(f"{args.train_output_path}/synthetic_data.csv", 
                    f"{metadata['name']}/synthetic_data.csv")
        #leaderboard.csv
        shutil.copy(f"{args.train_output_path}/leaderboard.csv", 
                    f"{metadata['name']}/leaderboard.csv")
        #config.yaml
        shutil.copy(f"{args.train_output_path}/.hydra/config.yaml", 
                    f"{metadata['name']}/config.yaml")
        #graph.graphml
        shutil.copy(f"{args.train_output_path}/graph.graphml",
                    f"{metadata['name']}/graph.graphml")

    # compress folder and double zip
    zipfile = double_zip_folder(metadata['name'], metadata['name'])
    
    # upload to dataverse
    data_description = f"""
        This is a synthetic dataset generated for the space project.
        The dataset was generated using the following configuration:
        base dataset: ## TODO: add base_name to metadata.yaml
        outcome: {metadata['predicted_outcome']}
        treatment: {metadata['treatment']}
        """
    
    upload_dataverse_data(
        zipfile,
        data_description,
        args.dataverse_baseurl, 
        args.dataverse_pid,
        args.dataverse_token
    )
    
    os.remove(zipfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_output_path', type=str)
    #parser.add_argument('--train_output_path', type=str, default='outputs/2023-05-25/18-57-12')
    parser.add_argument('--dataverse_baseurl', type=str, default='https://dataverse.harvard.edu')
    parser.add_argument('--dataverse_pid', type=str, default='doi:10.7910/DVN/SYNPBS')
    parser.add_argument('--dataverse_token', type=str, default=os.environ['DATAVERSE_TOKEN'])
    args = parser.parse_args()
    main(args)
