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
MAX_REQUEST_SIZE = 50331648 / 2 ## FIXME: This is a temporary fix to avoid exceeding the request size limit
MAX_BANDS = 4
BYTES_PER_PIXEL = 1
REQUEST_LIMIT = 5500  # Set the request limit to 5500 to avoid rate limiting

if len(sys.argv) < 4:
    print("Usage: python download_naip_by_states.py <state_names:str> <num_workers:int> <out_dir:str>")
    sys.exit(1)

# Get the sys arguments
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


def check_quota(lock, start_time, request_counter):
    with lock:
        # Get the current time
        current_time = time.time()

        # Calculate the elapsed time since the start of the current minute
        elapsed_time = current_time - start_time.value

        # If a minute has passed, reset the counter and timestamp
        if elapsed_time >= 60:
            request_counter.value = 0
            start_time.value = current_time

        # Check if the request counter has reached the quota limit
        if request_counter.value >= REQUEST_LIMIT:
            # Calculate the remaining time until the next minute
            remaining_time = 60 - elapsed_time
            logging.warning(f"Quota limit reached. Pausing for {remaining_time:.2f} seconds...")
            time.sleep(remaining_time)
            # Reset the counter and timestamp after pausing
            request_counter.value = 0
            start_time.value = time.time()


def initialize_ee():
    """Initialize the Earth Engine module."""
    ee.Initialize(url='https://earthengine-highvolume.googleapis.com', project='ee-zack')


def split_geometry(geometry, max_dim, scale, max_request_size, lock, start_time, request_counter):
    """Check if the geometry exceeds GEE maximum dimensions and 
    split it into smaller subregions if needed."""

    # Check quota before making a request
    check_quota(lock, start_time, request_counter)
    bbox = geometry.bounds().getInfo()['coordinates'][0]
    with lock:
        request_counter.value += 1

    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]

    width = max_x - min_x
    height = max_y - min_y

    width_m = width * 111320  # Approximate conversion factor for longitude
    height_m = height * 110540  # Approximate conversion factor for latitude

    num_splits = 1
    while (width_m / num_splits) / scale > max_dim or (height_m / num_splits) / scale > max_dim:
        num_splits *= 2

    # Calculate the size of each sub-region in pixels
    sub_region_width_m = width_m / num_splits
    sub_region_height_m = height_m / num_splits
    sub_region_pixels = (sub_region_width_m / scale) * (sub_region_height_m / scale)  # Number of pixels

    # Convert the size of each sub-region from pixels to bytes
    sub_region_bytes = sub_region_pixels * 4  # 4 bytes per pixel (assuming 4 bands, 1 byte per band)

    while sub_region_bytes > max_request_size:
        num_splits *= 2
        sub_region_width_m = width_m / num_splits
        sub_region_height_m = height_m / num_splits
        sub_region_pixels = (sub_region_width_m / scale) * (sub_region_height_m / scale)
        sub_region_bytes = sub_region_pixels * 4

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
        return [[[min_x, min_y], [max_x, max_y]]]


def getRequests(state_name: str, batch_size: int = 10, retries: int = 3, delay: int = 5, lock=None, start_time=None, request_counter=None):
    """Split counties in a state into subregions and return a list of 
    tuples with the county name followed by its subregions."""

    try:
        # Check quota before making a request
        check_quota(lock, start_time, request_counter)
        counties = ee.FeatureCollection('TIGER/2018/Counties').filter(
            ee.Filter.eq(
                'STATEFP',
                ee.FeatureCollection('TIGER/2018/States')
                .filter(ee.Filter.eq('NAME', state_name))
                .first()
                .get('STATEFP')
            )
        )
        with lock:
            request_counter.value += 4

        county_names = counties.aggregate_array('NAME').getInfo()
        with lock:
            request_counter.value += 1
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
                            max_request_size = MAX_REQUEST_SIZE,
                            lock = lock,
                            start_time = start_time,
                            request_counter = request_counter
                        )
                        with lock:
                            request_counter.value += 3
                        
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


def process_sub_region(state_name, county_name, sub_region_idx, sub_region, failed_sub_regions, lock, start_time, request_counter):
    """Download the sub-region images for each county in the list of tuples"""

    # Check quota before making a request
    check_quota(lock, start_time, request_counter)
    sub_region = ee.Geometry.Rectangle(sub_region)
    with lock:
        request_counter.value += 1

    sub_region_image = ee.ImageCollection('USDA/NAIP/DOQQ') \
                .filterBounds(sub_region) \
                .sort('system:time_start', False) \
                .select(['R', 'G', 'B', 'N']) \
                .first()

    if sub_region_image is None:
        logging.info(f"No image found for sub-region {sub_region_idx} in county {county_name} -- skipping")
        failed_sub_regions.append((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
        return

    sub_region_image = sub_region_image.clip(sub_region)

    if 'N' not in sub_region_image.bandNames().getInfo():
        logging.info(f"Skipping sub-region {sub_region_idx} in county {county_name} as it contains no bands")
        failed_sub_regions.append((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
        return

    with lock:
        request_counter.value += 7  # Increment the request counter by 7 to account for all preceding requests

    max_retries = 5
    backoff_factor = 1

    for attempt in range(max_retries):
        try:
            # Check quota before making a request
            check_quota(lock, start_time, request_counter)

            url = sub_region_image.getDownloadURL({
                'scale': SCALE,
                'format': 'GEO_TIFF',
                'region': sub_region.getInfo()['coordinates']
            })
            with lock:
                request_counter.value += 1

            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                r.raise_for_status()

            # Increment the request counter for the requests.get call
            with lock:
                request_counter.value += 1

            county_dir = os.path.join(out_dir, state_name, str(county_name))
            os.makedirs(county_dir, exist_ok=True)

            filename = f"{county_name}_{sub_region_idx}_naip.tif"
            filepath = os.path.join(county_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"Saved {filename} to {county_dir}")
            return  # Exit the function if successful

        except requests.exceptions.HTTPError as e:
            if r.status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTP error occurred: {e}")
                break
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error occurred: {e}. Retrying...")
            time.sleep(backoff_factor * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error occurred: {e}")
            break
        except ee.EEException as e:
            logging.error(f"Error processing sub_region {sub_region_idx}: {e}")
            break
        except Exception as e:
            logging.error(f"Unexpected error in sub_region {sub_region_idx}: {e}")
            break

    else:
        logging.error(f"Failed to download sub-region {sub_region_idx} for county {county_name} after {max_retries} attempts")
        failed_sub_regions.append((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))


if __name__ == '__main__':
    # Initialize request counter and timestamp using a multiprocessing manager
    quota_manager = multiprocessing.Manager()
    request_counter = quota_manager.Value('i', 0)
    start_time = quota_manager.Value('d', time.time())
    lock = quota_manager.Lock()  # Use Manager to create the lock

    # Initialize Earth Engine
    initialize_ee()

    failed_sub_regions_manager = multiprocessing.Manager()
    failed_sub_regions = failed_sub_regions_manager.list()

    program_start_time = time.time()

    try:
        for state_name in state_names:
            counties_and_sub_regions = getRequests(state_name, lock=lock, start_time=start_time, request_counter=request_counter)
            print(f'Found {len(counties_and_sub_regions)} counties in {state_name}')
            
            for index in range(len(counties_and_sub_regions)):
                county_name = counties_and_sub_regions[index][0]
                county_folder = os.path.join(out_dir, state_name, county_name)
                
                # Check if the county folder exists and contains the expected number of sub-region files
                if os.path.exists(county_folder):
                    expected_files = len(counties_and_sub_regions[index]) - 1  # Subtract 1 for the county name
                    existing_files = len([name for name in os.listdir(county_folder) if name.endswith('.tif')])
                    if existing_files == expected_files:
                        print(f'Skipping {county_name} County as it already exists and is complete.')
                        continue

                print(f'Beginning {county_name} County')
                county_start_time = time.time()

                tasks = getResults(index, counties_and_sub_regions, failed_sub_regions, state_name)
                
                # Create a multiprocessing pool with num_workers
                pool = multiprocessing.Pool(num_workers, initializer=initialize_ee)

                # Use starmap to parallelize the processing of sub-regions for the current county
                results = pool.starmap_async(process_sub_region, [(task[0], task[1], task[2], task[3], task[4], lock, start_time, request_counter) for task in tasks])
                    
                while not results.ready():
                    if time.time() - county_start_time > 1200:
                        logging.warning(f"Processing county {county_name} timed out.")
                        pool.terminate()
                        pool.join()
                        for task in tasks:
                            failed_sub_regions.append((task[1], task[2], task[3]))
                        break
                    time.sleep(1)
                
                if results.ready():
                    pool.close()
                    pool.join()
                    # Check for any failed tasks
                    for task, result in zip(tasks, results.get()):
                        if result is None:
                            failed_sub_regions.append((task[1], task[2], task[3]))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Save the failed sub-regions for the state to a text file
        failed_sub_regions_file = f"{os.path.join(out_dir, state_name)}/failed_sub_regions_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(failed_sub_regions_file, 'w') as f:
            for item in failed_sub_regions:
                f.write(f"{item}\n")

        program_end_time = time.time()

        print(f"Program finished in {program_end_time - program_start_time:.2f} seconds")
        print(f"Failed sub-regions saved to {failed_sub_regions_file}")