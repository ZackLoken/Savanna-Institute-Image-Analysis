import ee
import sys
import os
import requests
import logging
import multiprocessing
from multiprocessing import current_process
from retry import retry
import time


# set args for terminal processing (state name and number of workers)
state_name = sys.argv[1]
num_workers = int(sys.argv[2])


# Initialize the Earth Engine module to use the high volume endpoint (use whenever making automated requests)
ee.Initialize(url='https://earthengine-highvolume.googleapis.com', project='ee-zack')


# Configure logging of process information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)


def check_and_split_county(county_geometry, max_dim, scale, max_request_size):
    """Check if the geometry exceeds GEE maximum dimensions and 
    split it into smaller subregions if needed."""
    # Get the bounding box of the geometry
    bounds = county_geometry.bounds().getInfo()['coordinates'][0]
    min_x, min_y = bounds[0]
    max_x, max_y = bounds[2]
    
    width = max_x - min_x
    height = max_y - min_y

    # Convert degrees to meters (approx)
    width_m = width * 111319.5
    height_m = height * 111319.5
    
    # Calculate the number of splits needed
    num_splits = 1
    while (width_m / num_splits) / scale > max_dim or (height_m / num_splits) / scale > max_dim:
        num_splits *= 2

    # Further split if the request size exceeds the limit
    while (width_m * height_m * 4 / (num_splits ** 2)) > max_request_size:
        num_splits *= 2

    if num_splits > 1:
        x_step = width / num_splits
        y_step = height / num_splits

        # Split the region into smaller subregions if needed
        sub_regions = [
            ee.Geometry.Rectangle([
                min_x + i * x_step,
                min_y + j * y_step,
                min_x + (i + 1) * x_step,
                min_y + (j + 1) * y_step
            ])
            for i in range(num_splits)
            for j in range(num_splits)
        ]
        
        return sub_regions
    else:
        return [county_geometry]


def getRequests(state_name:str):
    """Split counties in a state into subregions and return a list of 
    tuples with the county name followed by its subregions."""
    try:
        # Load the counties FeatureCollection using the state name to filter
        counties = ee.FeatureCollection('TIGER/2018/Counties').filter(
            ee.Filter.eq(
                'STATEFP',
                ee.FeatureCollection('TIGER/2018/States')
                .filter(ee.Filter.eq('NAME', state_name))
                .first()
                .get('STATEFP')
            )
        )   

        # Function to process each county name in the list
        def process_county(county_name):
            # Split the county into smaller subregions
            sub_regions = check_and_split_county(
                county_geometry = counties.filter(ee.Filter.eq('NAME', county_name)).first().geometry(), 
                max_dim = 32768, 
                scale = 1, 
                max_request_size = 50331648
            )
            # Create a tuple with the county name followed by its subregions
            return (county_name, *sub_regions)
        # Use map to process each county and create the list of subregions
        return list(map(process_county, counties.aggregate_array('NAME').getInfo()))
    except Exception as e:
        logging.error(f'Error processing state {state_name}: {e}')
        return []


def getResults(index, county_sub_regions):
    """Download the sub-region images for each county in the list of tuples"""
    county_name, *sub_regions = county_sub_regions[index]
    logging.info(f'Processing {county_name}...')

    # create a directory for the county
    county_dir = f"C:/Users/exx/Desktop/CRP_Windbreak_ID/meta_chm_data/{state_name}" + '/' + str(county_name)
    os.makedirs(county_dir, exist_ok=True)

    start_time = time.time()

    # download image for each sub region in county
    def download_and_save(sub_region_idx_sub_region):
        sub_region_idx, sub_region = sub_region_idx_sub_region
        # Get the sub-region image
        sub_region_image = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').filterBounds(sub_region).mosaic().clip(sub_region)
        # Get the download URL for the image
        url = sub_region_image.getDownloadURL({
            'scale': 1,
            'format': 'GEO_TIFF'
        })

        # Download the TIFF from URL
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        # save the image to the county directory with filename: {county_name}_{sub_region_idx}_chm.tif
        with open(f'{county_dir}/{county_name}_{sub_region_idx}_chm.tif', 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f'Saved {county_name}_{sub_region_idx}_chm.tif to {county_dir} in {time.time() - start_time:.2f} seconds')

    list(map(download_and_save, enumerate(sub_regions)))


# configure logging and start multiprocessing
if __name__ == '__main__':
    program_start_time = time.time()

    # store counties and subregions in a list
    counties_and_sub_regions = getRequests(state_name)
    
    # create multiprocessing pool with 48 workers
    pool = multiprocessing.Pool(num_workers)

    # use starmap to parallelize the download process
    pool.starmap(getResults, enumerate(counties_and_sub_regions))

    # close and join the pool to ensure that all processes are completed before exiting
    pool.close()
    pool.join()

    program_end_time = time.time()

    print(f"Program finished in {program_end_time - program_start_time:.2f} seconds")