import argparse
import os
import shutil
import yaml

def main(args):
    # load metadata
    with open(f"{args.train_output_path}/metadata.yaml", 'r') as f:
        metadata = yaml.load(f, Loader=yaml.FullLoader)
      
    # create folder
    os.mkdir(metadata['base_name'])
    #metadata.yaml
    shutil.copy(f"{args.train_output_path}/metadata.yaml", 
                f"{metadata['base_name']}/metadata.yaml")
    #synthetic_data.csv
    shutil.copy(f"{args.train_output_path}/synthetic_data.csv", 
                f"{metadata['base_name']}/synthetic_data.csv")
    #leaderboard.csv
    shutil.copy(f"{args.train_output_path}/leaderboard.csv", 
                f"{metadata['base_name']}/leaderboard.csv")
    #config.yaml
    shutil.copy(f"{args.train_output_path}/.hydra/config.yaml", 
                f"{metadata['base_name']}/config.yaml")
    #graph.graphml
    shutil.copy(f"{args.train_output_path}/graph.graphml",
                f"{metadata['base_name']}/graph.graphml")
    
    # compress folder
    shutil.make_archive(metadata['base_name'], 'zip', metadata['base_name'])
    ## shutil.rmtree(metadata['base_name']) ## removes folder
    
    ## add code to upload to dataverse (already in space repository)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_output_path', type=str)
    ## parser.add_argument('--train_output_path', type=str, default='outputs/2023-05-19/22-39-05')
    args = parser.parse_args()
    main(args)
