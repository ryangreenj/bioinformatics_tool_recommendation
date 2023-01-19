import os
import requests
import time

from src import constants

def download(source):
    # Download all public/shared workflows from the given usegalaxy source
    out_dir = os.path.join(constants.RAW_GALAXY_FILES, source.replace(".", "_"))
    os.makedirs(out_dir, exist_ok=True)
    
    # Don't download workflows that are already cached
    existing_files = [os.path.splitext(file)[0] for file in os.listdir(out_dir)]
    
    # Get list of workflows from server
    workflow_list_url = f"https://{source}/api/workflows"
    print("Downloading workflow list from ", workflow_list_url)
    workflow_list = requests.get(workflow_list_url).json()
    
    to_download = []
    for workflow in workflow_list:
        if workflow["id"] not in existing_files:
            to_download.append(workflow["id"])
    
    print(f"Downloading {len(to_download)} workflows from {source}")
    
    # Download with respect to the API wait time specified in constants
    prev_time = 0
    num = 0
    for workflow_id in to_download:
        num += 1
        
        download_url = f"https://{source}/api/workflows/{workflow_id}/download"
        
        while (time.time() - prev_time < constants.GALAXY_API_WAIT):
            time.sleep(0.1)
        
        prev_time = time.time()
        print(f"Downloading {num}/{len(to_download)}: {workflow_id}")
        r = requests.get(download_url)
        
        if r.status_code != 200:
            print(f"Error downloading {workflow_id}")
            print("Skipping...")
            continue
        
        with open(os.path.join(out_dir, f"{workflow_id}.ga"), "w") as f:
            f.write(r.text)

def run():
    print("Downloading workflows from AllGalaxy sources")
    with open(constants.WORKFLOW_SOURCES) as f:
        sources = f.read().splitlines()
    
    for source in sources:
        download(source)
    
    print("Done downloading workflows")