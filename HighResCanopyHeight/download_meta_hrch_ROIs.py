import ee
import os
import shutil
import requests
from requests.exceptions import HTTPError
import logging
import multiprocessing
from multiprocessing import current_process
from retry import retry
import time


# initialize the Earth Engine module to use the high volume endpoint (use whenever making automated requests)
ee.Initialize(url = 'https://earthengine-highvolume.googleapis.com', project = 'ee-zack')


# Configure logging of process information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)


def check_and_split_region(geometry, max_dim, scale, max_request_size):
    # Get the bounding box of the geometry
    bounds = geometry.bounds().getInfo()['coordinates'][0]
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
        return [geometry]


def getRequests():
    """Generate a list of work items to be downloaded.
    
    Extract the county_codes from the TIGER/2018/counties dataset as work units
    """
    try:
        # Load Illinois and Iowa counties into one feature collection
        counties = ee.FeatureCollection('TIGER/2018/Counties').filter(
            ee.Filter.Or(ee.Filter.eq('STATEFP', '17'), ee.Filter.eq('STATEFP', '19'))
        )
        
        # Get the list of county names
        county_names = counties.aggregate_array('NAME').getInfo()
        if not county_names:
            logging.error("No county names found")
            return []

        # Function to process each county
        def process_county(county_name):
            county = counties.filter(ee.Filter.eq('NAME', county_name)).first()
            county_geometry = county.geometry()
            
            # Split the county into smaller subregions
            sub_regions = check_and_split_region(county_geometry, max_dim=32768, scale=1, max_request_size=50331648)
            
            # Create a tuple with the county name followed by its subregions
            return (county_name, *sub_regions)
        
        # Use map to process each county and create the list of subregions
        subregions_list = list(map(process_county, county_names))
        
        return subregions_list
    except Exception as e:
        logging.error(f"Error in getRequests: {e}")
        return []


def process_subregion(county_name, sub_region, idx, dataset, out_dir):
    """Process a single subregion"""
    start_time = time.time()
    try:
        result = dataset.mosaic().getDownloadURL({
            'scale': 1,
            'region': sub_region,
            'format': 'GEO_TIFF'
        })

        # Download the TIFF from URL
        r = requests.get(result, stream=True)
        if r.status_code != 200:
            r.raise_for_status()
        
        # Save the image to the county directory
        out_path = os.path.join(out_dir, f"{county_name}_CHM_{idx}.tif")
        with open(out_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        logging.info(f"Saved {county_name}_CHM_{idx}.tif to {out_path} in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logging.error(f"Error processing subregion {idx} for {county_name}: {e}")


@retry(tries=10, delay=3, backoff=2)
def getResult(index, county_name, *sub_regions):
    """Handle HTTP requests to download one result"""

    process_name = current_process().name
    logging.info(f"Process {process_name} - Starting item {index} - {county_name}")

    county = (ee.FeatureCollection('TIGER/2018/Counties')
          .filter(ee.Filter.Or(ee.Filter.eq('STATEFP', '17'), # Illinois
                               ee.Filter.eq('STATEFP', '19'))) # Iowa
          .filter(ee.Filter.eq('NAME', county_name)))
    
    # Determine the state FIPS code based on the county name
    county_info = county.first().getInfo()
    if county_info is None:
        logging.error(f"County info for {county_name} is None")
        return
    state_fips = county_info['properties']['STATEFP']

    # Set output directory based on the state FIPS code
    out_dir = f"C:/Users/exx/Documents/GitHub/Savanna-Institute/HighResCanopyHeight/chm_data/{'illinois' if state_fips == '17' else 'iowa'}/{county_name}"
    os.makedirs(out_dir, exist_ok=True)
    
    def metaHRCH(image):
        # clip image collection to county
        return image.clip(county.geometry()).rename('CHM').set('county', county_name)
    
    # Pull meta image collection and clip to counties
    dataset = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').map(metaHRCH)

    # Prepare arguments for processing subregions
    args = [(county_name, sub_region, idx, dataset, out_dir) for idx, sub_region in enumerate(sub_regions)]

    # Use map to process each subregion
    list(map(lambda arg: process_subregion(*arg), args))

    logging.info(f"Process {process_name} finished item {index} for {county_name}")


# program start time
program_start_time = time.time()

# configure logging and start multiprocessing
if __name__ == '__main__':
    # get list of work items
    work_items = getRequests()
    if not work_items:
        logging.error("No items to process")
    else:
        # create multiprocessing pool with 48 workers
        pool = multiprocessing.Pool(48)

        # use starmap to parallelize the download process
        pool.starmap(getResult, [(index, work_item[0], *work_item[1:]) for index, work_item in enumerate(work_items)])

        # close and join the pool to ensure that all processes are completed before exiting
        pool.close()
        pool.join()

program_end_time = time.time()

print(f"Program finished in {program_end_time - program_start_time:.2f} seconds")