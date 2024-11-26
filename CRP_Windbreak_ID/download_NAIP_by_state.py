import ee
import sys
import os
import requests
import logging
import multiprocessing
import time

# Constants
MAX_DIM = 32768
SCALE = 0.6
MAX_REQUEST_SIZE = 50331648
MAX_BANDS = 4
BYTES_PER_PIXEL = 1

# Ensure the out_dir argument is provided
if len(sys.argv) < 4:
    print("Usage: python download_naip_by_states.py <state_names:str> <num_workers:int> <out_dir:str>")
    sys.exit(1)

# Get the out_dir argument
state_names = sys.argv[1].split(',')
num_workers = int(sys.argv[2])
out_dir = os.path.abspath(sys.argv[3])

# Create the directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# Configure logging of process information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

def initialize_ee():
    """Initialize the Earth Engine module."""
    ee.Initialize(url='https://earthengine-highvolume.googleapis.com', project='ee-zack-loken')

initialize_ee()

def split_geometry(geometry, max_dim, scale, max_request_size):
    """Check if the geometry exceeds GEE maximum dimensions and 
    split it into smaller subregions if needed."""
    bbox = geometry.bounds().getInfo()['coordinates'][0]
    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]
    
    width = max_x - min_x
    height = max_y - min_y

    width_m = width * 111319.5
    height_m = height * 111319.5
    
    num_splits = 1
    while (width_m / num_splits) / scale > max_dim or (height_m / num_splits) / scale > max_dim:
        num_splits *= 2

    while (width_m * height_m * 4) > max_request_size:
        num_splits *= 4

    if num_splits > 1:
        x_step = width / num_splits
        y_step = height / num_splits

        sub_regions = [
            [
                min_x + i * x_step,
                min_y + j * y_step,
                min_x + (i + 1) * x_step,
                min_y + (j + 1) * y_step
            ]
            for i in range(num_splits)
            for j in range(num_splits)
        ]
        logging.info(f"Split into {len(sub_regions)} sub-regions.")
        return sub_regions
    else:
        return [geometry]

def getRequests(state_name: str, batch_size: int = 10, retries: int = 3, delay: int = 5):
    """Split counties in a state into subregions and return a list of 
    tuples with the county name followed by its subregions."""
    try:
        counties = ee.FeatureCollection('TIGER/2018/Counties').filter(
            ee.Filter.eq(
                'STATEFP',
                ee.FeatureCollection('TIGER/2018/States')
                .filter(ee.Filter.eq('NAME', state_name))
                .first()
                .get('STATEFP')
            )
        )

        county_names = counties.aggregate_array('NAME').getInfo()
        all_results = []

        for i in range(0, len(county_names), batch_size):
            batch = county_names[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1} of {len(county_names) // batch_size + 1}")

            batch_results = []
            for county_name in batch:
                attempt = 0
                while attempt < retries:
                    try:
                        logging.info(f"Loading county: {county_name}")
                        sub_regions = split_geometry(
                            geometry = counties.filter(ee.Filter.eq('NAME', county_name)).first().geometry(), 
                            max_dim = MAX_DIM, 
                            scale = SCALE, 
                            max_request_size = MAX_REQUEST_SIZE
                        )
                        
                        batch_results.append((county_name, *sub_regions))
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt < retries:
                            time.sleep(delay * attempt)
                        else:
                            logging.error(f"Failed to process county {county_name} after {retries} attempts.")
                            continue

            all_results.extend(batch_results)
            time.sleep(1)

        return all_results
    
    except Exception as e:
        logging.error(f'Error processing state {state_name}: {e}')
        return []

def getResults(index, counties_and_sub_regions, failed_sub_regions, state_name):
    """Prepare sub-region processing tasks for each county"""
    county_name, *sub_regions = counties_and_sub_regions[index]

    tasks = [(state_name, county_name, sub_region_idx, sub_region_geojson, failed_sub_regions) for sub_region_idx, sub_region_geojson in enumerate(sub_regions)]
    
    return tasks

def process_sub_region(state_name, county_name, sub_region_idx, sub_region, failed_sub_regions):
    """Download the sub-region images for each county in the list of tuples"""
    sub_region = ee.Geometry.Rectangle(sub_region)

    sub_region_image = ee.ImageCollection('USDA/NAIP/DOQQ') \
                .filterBounds(sub_region) \
                .filter(ee.Filter.calendarRange(2018, 2024, 'year')) \
                .sort('system:time_start', False) \
                .first() \
                .clip(sub_region)
    
    max_retries = 5
    backoff_factor = 1

    for attempt in range(max_retries):
        try:
            url = sub_region_image.getDownloadURL({
                'scale': SCALE,
                'format': 'GEO_TIFF',
                'bands': ['R', 'G', 'B', 'N']
            })

            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                r.raise_for_status()

            county_dir = os.path.join(out_dir, state_name, str(county_name))
            os.makedirs(county_dir, exist_ok=True)

            filename = f"{county_name}_{sub_region_idx}_naip.tif"
            filepath = os.path.join(county_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Saved {filename} to {county_dir}")
            break

        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTP error occurred: {e}")
                failed_sub_regions.put((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
                break
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error occurred: {e}")
            failed_sub_regions.put((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
            break
        except ee.EEException as e:
            logging.error(f"Error processing sub_region {sub_region_idx}: {e}")
            failed_sub_regions.put((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
            break
        except Exception as e:
            logging.error(f"Unexpected error in sub_region {sub_region_idx}: {e}")
            failed_sub_regions.put((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
            break

    else:
        logging.error(f"Failed to download sub-region {sub_region_idx} for county {county_name} after {max_retries} attempts")
        failed_sub_regions.put((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    failed_sub_regions = manager.list()

    program_start_time = time.time()

    for state_name in state_names:
        counties_and_sub_regions = getRequests(state_name)
        print(f'Found {len(counties_and_sub_regions)} counties in {state_name}')
        
        for index in range(len(counties_and_sub_regions)):
            county_folder = os.path.join(out_dir, state_name, counties_and_sub_regions[index][0])
            if os.path.exists(county_folder):
                print(f'Skipping {counties_and_sub_regions[index][0]} County as it already exists.')
                continue

            print(f'Beginning {counties_and_sub_regions[index][0]} County')
            county_start_time = time.time()

            tasks = getResults(index, counties_and_sub_regions, failed_sub_regions, state_name)
            
            # Create a multiprocessing pool with num_workers
            pool = multiprocessing.Pool(num_workers, initializer=initialize_ee)

            # Use starmap to parallelize the processing of sub-regions for the current county
            results = pool.starmap_async(process_sub_region, tasks)
                
            while not results.ready():
                if time.time() - county_start_time > 1200:
                    logging.warning(f"Processing county {counties_and_sub_regions[index][0]} timed out.")
                    pool.terminate()
                    pool.join()
                    for task in tasks[results._index:]:
                        failed_sub_regions.append((task[1], task[2], task[3]))
                    break
                time.sleep(1)
            
            if results.ready():
                pool.close()
                pool.join()

        failed_sub_regions_file = f"{os.path.join(out_dir, state_name)}/failed_sub_regions.txt"
        with open(failed_sub_regions_file, 'w') as f:
            for item in failed_sub_regions:
                f.write(f"{item}\n")

    program_end_time = time.time()

    print(f"Program finished in {program_end_time - program_start_time:.2f} seconds")
    print(f"Failed sub-regions saved to {failed_sub_regions_file}")