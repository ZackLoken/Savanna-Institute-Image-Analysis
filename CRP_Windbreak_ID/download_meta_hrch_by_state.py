import ee
import sys
import os
import numpy as np
import requests
import logging
import multiprocessing
import time

# Constants
MAX_DIM = 32768
MAX_REQUEST_SIZE = 50331648
MAX_BANDS = 1 # assuming 1 byte per band for a pixel
REQUEST_LIMIT = 5500  # quota

if len(sys.argv) < 4:
    print("Usage: python download_meta_hrch_by_state.py <state_names:str> <num_workers:int> <out_dir:str>")
    print('Example: python download_meta_hrch_by_state.py "Iowa,Illinois" 24 "/path/to/output"')
    sys.exit(1)

# Get the sys arguments
state_names = sys.argv[1].split(',')
num_workers = int(sys.argv[2])
out_dir = os.path.abspath(sys.argv[3])
os.makedirs(out_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

def check_quota(lock, start_time, request_counter):
    with lock:
        current_time = time.time()
        elapsed_time = current_time - start_time.value
        if elapsed_time >= 60:
            request_counter.value = 0
            start_time.value = current_time
        if request_counter.value >= REQUEST_LIMIT:
            remaining_time = 60 - elapsed_time
            logging.warning(f"Quota limit reached. Pausing for {remaining_time:.2f} seconds...")
            time.sleep(remaining_time)
            request_counter.value = 0
            start_time.value = time.time()

def initialize_ee():
    try:
        ee.Authenticate()
        ee.Initialize(url='https://earthengine-highvolume.googleapis.com', project='ee-zack')
        logging.info("Earth Engine initialized successfully")
    except Exception as e:
        logging.error(f"Earth Engine initialization failed: {e}")
        raise e  # Re-raise to ensure the error is handled properly

def split_geometry(geometry, max_dim, scale, max_request_size, max_bands, lock, start_time, request_counter):
    check_quota(lock, start_time, request_counter)
    bbox = geometry.bounds().getInfo()['coordinates'][0]
    with lock:
        request_counter.value += 1

    min_x, min_y = bbox[0]
    max_x, max_y = bbox[2]
    width = max_x - min_x
    height = max_y - min_y

    # Simple conversion
    width_m = width * 111319.5
    height_m = height * 111319.5  

    num_splits = 1
    while (width_m / num_splits) / scale > max_dim or (height_m / num_splits) / scale > max_dim:
        num_splits *= 2

    # Calculate without overlap first to determine splits
    pixels_width = width_m / scale
    pixels_height = height_m / scale
    bytes_per_pixel = max_bands * 1.2 
    while (pixels_width * pixels_height * bytes_per_pixel / (num_splits ** 2)) > max_request_size:
        num_splits *= 2

    if num_splits > 1:
        x_step = width / num_splits
        y_step = height / num_splits
        
        # NO overlap - simple grid
        sub_regions = []
        for j in range(num_splits):
            for i in range(num_splits):
                sub_regions.append([
                    min_x + i * x_step,
                    min_y + j * y_step,
                    min_x + (i + 1) * x_step,
                    min_y + (j + 1) * y_step
                ])
        
        logging.info(f"Split into {len(sub_regions)} sub-regions.")
        return sub_regions
    else:
        return [[[min_x, min_y], [max_x, max_y]]]

def getRequests(state_name: str, batch_size: int = 10, retries: int = 3, delay: int = 5, lock=None, start_time=None, request_counter=None):
    try:
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

        # Determine scale from first county
        state_scale = 1  # Default scale if we can't determine it
        if county_names:
            first_county = counties.filter(ee.Filter.eq('NAME', county_names[0])).first()
            try:
                check_quota(lock, start_time, request_counter)
                # Get a sample image from the first county
                sample_image = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight') \
                    .filterBounds(first_county.geometry()) \
                    .first()
                
                if sample_image is not None:
                    projection_info = sample_image.projection().getInfo()
                    state_scale = abs(projection_info['transform'][0])
                    logging.info(f"Determined scale for {state_name}: {state_scale} meters")
                    with lock:
                        request_counter.value += 3
            except Exception as e:
                logging.warning(f"Error determining scale for {state_name}: {e}. Using default scale.")

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
                            scale = state_scale,  # Use scale determined from first county
                            max_request_size = MAX_REQUEST_SIZE,
                            max_bands = MAX_BANDS,
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
    county_name, *sub_regions = counties_and_sub_regions[index]
    tasks = [(state_name, county_name, sub_region_idx, sub_region_geojson, failed_sub_regions)
             for sub_region_idx, sub_region_geojson in enumerate(sub_regions)]
    return tasks

def process_sub_region(state_name, county_name, sub_region_idx, sub_region, failed_sub_regions, lock, start_time, request_counter):
    check_quota(lock, start_time, request_counter)
    sub_region = ee.Geometry.Rectangle(sub_region)
    with lock:
        request_counter.value += 1

    sub_region_image = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight') \
                    .filterBounds(sub_region) \
                    .first()

    if sub_region_image is None:
        logging.info(f"Skipping sub-region {sub_region_idx} in county {county_name} -- no image found")
        return False

    try:
        sub_region_image_info = sub_region_image.getInfo()
        
        # Get the projection information and extract the scale (resolution)
        projection_info = sub_region_image.projection().getInfo()
        # scale_x (index 0) is typically the resolution in meters
        image_scale = abs(projection_info['transform'][0])
        logging.info(f"CHM image for {county_name} sub-region {sub_region_idx} has resolution: {image_scale} meters")
        
    except Exception as e:
        logging.error(f"Error retrieving info for sub-region {sub_region_idx} in county {county_name}: {e}")
        return False

    if sub_region_image_info is None:
        logging.info(f"Skipping sub-region {sub_region_idx} in county {county_name} -- no valid info")
        return False

    sub_region_image = sub_region_image.clip(sub_region) 

    with lock:
        request_counter.value += 5  # Increment by 6 to account for preceeding requests to gee

    max_retries = 5
    backoff_factor = 1
    for attempt in range(max_retries):
        try:
            check_quota(lock, start_time, request_counter)
            url = sub_region_image.getDownloadURL({
                'scale': image_scale,  # Use the dynamically determined scale
                'format': 'GEO_TIFF',
            })
            with lock:
                request_counter.value += 1
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                r.raise_for_status()
            with lock:
                request_counter.value += 1
            county_dir = os.path.join(out_dir, state_name, str(county_name))
            os.makedirs(county_dir, exist_ok=True)
            filename = f"{county_name}_{sub_region_idx}_chm.tif"
            filepath = os.path.join(county_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Saved {filename} to {county_dir}")
            return True
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

    logging.error(f"Failed to download sub-region {sub_region_idx} for county {county_name} after {max_retries} attempts")
    failed_sub_regions.append((county_name, sub_region_idx, sub_region.getInfo()['coordinates']))
    return False

if __name__ == '__main__':
    quota_manager = multiprocessing.Manager()
    request_counter = quota_manager.Value('i', 0)
    start_time = quota_manager.Value('d', time.time())
    lock = quota_manager.Lock()

    initialize_ee()

    # Global failed list across states (if desired)
    global_failed = multiprocessing.Manager().list()

    program_start_time = time.time()

    try:
        for state_name in state_names:
            # For each state, use a local failed list
            state_failed = multiprocessing.Manager().list()

            counties_and_sub_regions = getRequests(state_name, lock=lock, start_time=start_time, request_counter=request_counter)
            print(f'Found {len(counties_and_sub_regions)} counties in {state_name}')
            
            for index in range(len(counties_and_sub_regions)):
                county_name = counties_and_sub_regions[index][0]
                county_folder = os.path.join(out_dir, state_name, county_name)
                expected_files = len(counties_and_sub_regions[index]) - 1
                if os.path.exists(county_folder):
                    existing_files = len([name for name in os.listdir(county_folder) if name.endswith('.tif')])
                    if existing_files == expected_files:
                        print(f'Skipping {county_name} County as it already exists and is complete.')
                        continue

                print(f'Beginning {county_name} County')
                county_start_time = time.time()
                tasks = getResults(index, counties_and_sub_regions, state_failed, state_name)
                pool = multiprocessing.Pool(num_workers, initializer=initialize_ee)
                results_async = pool.starmap_async(process_sub_region, 
                    [ (task[0], task[1], task[2], task[3], state_failed, lock, start_time, request_counter) for task in tasks ]
                )

                # Monitor processing time with timeout (e.g. 1 hour)
                timeout_seconds = 3600
                while not results_async.ready():
                    if time.time() - county_start_time > timeout_seconds:
                        logging.warning(f"Processing county {county_name} timed out.")
                        pool.terminate()
                        pool.join()
                        # Mark any pending tasks as failed
                        for task in tasks:
                            state_failed.append((task[1], task[2], task[3]))
                        break
                    time.sleep(1)

                if results_async.ready():
                    pool.close()
                    pool.join()
                    results_list = results_async.get()
                    for i, result in enumerate(results_list):
                        if not result:
                            state_failed.append((tasks[i][1], tasks[i][2], tasks[i][3]))
                    print(f'Finished {county_name} County')

            # Now reattempt failed sub-regions serially for this state
            max_state_retries = 3
            retry_attempt = 0
            while len(state_failed) > 0 and retry_attempt < max_state_retries:
                logging.info(f"Retrying {len(state_failed)} failed sub-regions for {state_name}, attempt {retry_attempt+1}...")
                current_failures = list(state_failed)
                # Clear the list before retries
                state_failed[:] = []
                for failure in current_failures:
                    county, sub_region_idx, sub_region_coords = failure
                    # Directly call process_sub_region (serially)
                    result = process_sub_region(state_name, county, sub_region_idx, sub_region_coords, state_failed, lock, start_time, request_counter)
                    if not result:
                        state_failed.append((county, sub_region_idx, sub_region_coords))
                retry_attempt += 1

            # Append any remaining failures to the global list
            for item in state_failed:
                global_failed.append((state_name, *item))
            # Save state-specific failed regions to file
            failed_sub_regions_file = os.path.join(out_dir, state_name, f"failed_sub_regions_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            with open(failed_sub_regions_file, 'w') as f:
                for item in state_failed:
                    f.write(f"{item}\n")
            print(f"{len(state_failed)} failed sub-regions remain for {state_name} after retrying.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        program_end_time = time.time()
        print(f"Program finished in {program_end_time - program_start_time:.2f} seconds")
        print(f"{len(global_failed)} failed sub-regions (global) saved.")