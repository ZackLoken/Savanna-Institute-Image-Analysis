import Metashape
import os
import gc
import sys
import ast
import time
import cv2
import numpy as np
import re
from tqdm import tqdm
import pyexiv2

def enable_multi_core():
    """Configure Metashape to use only GPU device 0 (not device 1)"""
    Metashape.app.cpu_enable = True
    Metashape.app.gpu_mask = 1
    
    devices = Metashape.app.enumGPUDevices()
    device_name = devices[0].get('name', 'GPU 0') if devices else "No GPU detected"
    
    print(f"Multi-core processing enabled. Using GPU {device_name} only")

def find_files(folder, valid_types):
    try:
        valid_types = [ext.lower() for ext in valid_types]
        return [os.path.join(folder, entry.name)
                for entry in os.scandir(folder)
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_types]
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
        return []

def process_rgb_images(input_folder, output_folder, debug=True):
    """Process RGB images with lens distortion and vignetting corrections using equations outlined in image processing guide for Mavic 3M"""
    os.makedirs(output_folder, exist_ok=True)
    processed_count = 0
    failed_count = 0
    
    # Get image files
    valid_exts = [".jpg", ".jpeg", ".tif", ".tiff"]
    photo_paths = find_files(input_folder, valid_exts)
    
    if not photo_paths:
        print("No image files found")
        return False
    
    # Process images sequentially
    for input_path in tqdm(photo_paths, desc="Processing", unit="img", ncols=80):
        try:
            filename = os.path.basename(input_path)
            output_path = os.path.join(output_folder, filename)
            
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                print(f"Error loading {filename}")
                failed_count += 1
                continue
                
            # Extract XMP metadata
            with open(input_path, 'rb') as f:
                img_bytes = f.read()
                xmp_start = img_bytes.find(b'<x:xmpmeta')
                xmp_end = img_bytes.find(b'</x:xmpmeta')
                
                xmp_data = img_bytes[xmp_start:xmp_end+12].decode('utf-8', errors='ignore') if xmp_start >= 0 and xmp_end > xmp_start else ""
            
            height, width = img.shape[:2]
            
            # Get parameters
            center_x_match = re.search(r'CalibratedOpticalCenterX="([^"]+)"', xmp_data)
            center_y_match = re.search(r'CalibratedOpticalCenterY="([^"]+)"', xmp_data)
            vignette_match = re.search(r'VignettingData="([^"]+)"', xmp_data)
            distortion_match = re.search(r'DewarpData="([^"]+)"', xmp_data)
            
            center_x = float(center_x_match.group(1)) if center_x_match else width/2
            center_y = float(center_y_match.group(1)) if center_y_match else height/2
            
            # Apply lens distortion correction
            if distortion_match:
                distortion_parts = distortion_match.group(1).split(';')
                if len(distortion_parts) > 1:
                    dist_params = np.array([float(x) for x in distortion_parts[1].split(',')])
                    
                    fx, fy = dist_params[0:2]
                    cx, cy = dist_params[2:4]
                    k1, k2, p1, p2, k3 = dist_params[4:9]
                    
                    camera_matrix = np.array([
                        [fx, 0, center_x + cx],
                        [0, fy, center_y + cy],
                        [0, 0, 1]
                    ])
                    dist_coeffs = np.array([k1, k2, p1, p2, k3])
                    
                    img = cv2.undistort(img, camera_matrix, dist_coeffs)
            
            # Apply vignetting correction with original code
            if vignette_match:
                vignette_coeffs = np.array([float(x) for x in vignette_match.group(1).split(',')])
            else:
                base_coeffs = np.array([-0.00001, -0.000005, -0.000001, -0.0000005, -0.0000001, -0.00000005])
                vignette_coeffs = base_coeffs * 0.7
            
            # Apply correction
            y, x = np.mgrid[0:height, 0:width]
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_radius = np.sqrt(width**2 + height**2) / 2
            r_normalized = r / max_radius
            
            correction = np.ones_like(r)
            for j, k in enumerate(vignette_coeffs):
                power = j + 1
                correction += k * r_normalized**power
            
            correction = np.clip(correction, 0.8, 1.2)
            
            for c in range(img.shape[2]):
                img[:,:,c] = np.clip(img[:,:,c] * correction, 0, 255).astype(np.uint8)
            
            # Save processed image
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Copy metadata
            try:
                source_meta = pyexiv2.Image(input_path)
                exif_data = source_meta.read_exif()
                xmp_data = source_meta.read_xmp()
                iptc_data = source_meta.read_iptc()
                source_meta.close()
                
                dest_meta = pyexiv2.Image(output_path)
                dest_meta.modify_exif(exif_data)
                dest_meta.modify_xmp(xmp_data)
                dest_meta.modify_iptc(iptc_data)
                dest_meta.close()
                
                processed_count += 1
            except Exception as e:
                print(f"Error copying metadata for {filename}: {e}")
                failed_count += 1
        except Exception as e:
            print(f"Error processing {os.path.basename(input_path)}: {str(e)}")
            failed_count += 1
    
    # Verify a sample
    if processed_count > 0 and debug:
        sample_file = os.path.join(output_folder, os.path.basename(photo_paths[0]))
        print(f"Verifying sample file: {sample_file}")
        
        try:
            orig_meta = pyexiv2.Image(photo_paths[0])
            proc_meta = pyexiv2.Image(sample_file)
            
            orig_exif_count = len(orig_meta.read_exif())
            proc_exif_count = len(proc_meta.read_exif())
            orig_xmp_count = len(orig_meta.read_xmp())
            proc_xmp_count = len(proc_meta.read_xmp())
            
            print(f"Original image: {orig_exif_count} EXIF tags, {orig_xmp_count} XMP tags")
            print(f"Processed image: {proc_exif_count} EXIF tags, {proc_xmp_count} XMP tags")
            
            orig_meta.close()
            proc_meta.close()
        except Exception as e:
            print(f"Error verifying metadata: {e}")
    
    print(f"Processed {processed_count} images with corrections")
    if failed_count > 0:
        print(f"Failed to process {failed_count} images")
    
    return processed_count > 0

def reset_region(chunk):
    """
    Reset the region and make it much larger than the points;
    necessary because if points go outside the region,
    they get clipped when saving
    """
    chunk.resetRegion()
    region_dims = chunk.region.size
    region_dims[2] *= 3 # Increase height by 3x
    chunk.region.size = region_dims
    print("Region reset to prevent point clipping.")
    return True

def adaptive_filter_reconstruction_uncertainty(chunk, initial_threshold=15, target_percentage=0.2):
    """
    Adaptively filter tie points based on reconstruction uncertainty
    """
    print("Filtering by Reconstruction Uncertainty (adaptive threshold)...")
    f = Metashape.TiePoints.Filter()
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ReconstructionUncertainty)
    total_points = len(chunk.tie_points.points)
    
    # Find threshold that removes specified percentage of points
    threshold = initial_threshold
    while (len([i for i in f.values if i >= threshold])/total_points) >= target_percentage:
        threshold += 0.1
    threshold = round(threshold, 1)
    
    print(f"Removing points with reconstruction uncertainty >= {threshold}")
    f.removePoints(threshold)
    return threshold

def adaptive_filter_projection_accuracy(chunk, initial_threshold=2, target_percentage=0.3):
    """
    Adaptively filter tie points based on projection accuracy
    """
    print("Filtering by Projection Accuracy (adaptive threshold)...")
    f = Metashape.TiePoints.Filter()
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ProjectionAccuracy)
    total_points = len(chunk.tie_points.points)
    
    # Find threshold that removes specified percentage of points
    threshold = initial_threshold
    while (len([i for i in f.values if i >= threshold])/total_points) >= target_percentage:
        threshold += 0.1
    threshold = round(threshold, 1)
    
    print(f"Removing points with projection accuracy >= {threshold}")
    f.removePoints(threshold)
    return threshold

def adaptive_filter_reprojection_error(chunk, initial_threshold=0.3, target_percentage=0.05):
    """
    Adaptively filter tie points based on reprojection error
    """
    print("Filtering by Reprojection Error (adaptive threshold)...")
    f = Metashape.TiePoints.Filter()
    f.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)
    total_points = len(chunk.tie_points.points)
    
    # Find threshold that removes specified percentage of points
    threshold = initial_threshold
    while (len([i for i in f.values if i >= threshold])/total_points) >= target_percentage:
        threshold += 0.005
    threshold = round(threshold, 2)
    
    print(f"Removing points with reprojection error >= {threshold}")
    f.removePoints(threshold)
    return threshold

class ProgressTimer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.last_printed_percentage = -5
    
    def update(self, p):
        if p - self.last_printed_percentage >= 5 or p >= 100:
            elapsed = float(time.time() - self.start_time)
            if p > 0:
                remaining_sec = (elapsed / p) * (100 - p)
                print('Progress: {:.0f}%, est. time left: {:.0f} sec'.format(p, remaining_sec))
            else:
                print('Progress: {:.0f}%, est. time left: unknown'.format(p))
            self.last_printed_percentage = p

progress_timer = ProgressTimer()

def main():
    """
    Main function to automate the reconstruction process of DJI Mavic 3M imagery in Agisoft Metashape.
    This script is designed to be run from the command line with a list of folder paths containing images.
    """
    if len(sys.argv) != 2:
        print("Usage: python automated_reconstruction.py \"['<images_folder1>', '<images_folder2>', ...]\"")
        sys.exit(1)

    # Enable multi-core processing and use GPU device 0
    enable_multi_core()

    try:
        folder_paths = ast.literal_eval(sys.argv[1])
        if not isinstance(folder_paths, list):
            raise ValueError("Provided argument is not a list.")
    except Exception as e:
        print(f"Error parsing argument: {e}")
        sys.exit(1)
        
    folder_paths = [os.path.normpath(p) for p in folder_paths]
    valid_exts = [".JPG", ".JPEG", ".TIF", ".TIFF"]

    for folder in folder_paths:
        try:
            if not os.path.isdir(folder):
                print(f"Warning: {folder} is not a valid folder. Skipping.")
                continue
            
            photos = find_files(folder, valid_exts)
            if not photos:
                print(f"No valid image files found in folder {folder}. Skipping.")
                continue
            print(f"Processing folder: {folder} ({len(photos)} photos found).")
            
            parent_folder = os.path.basename(os.path.dirname(os.path.normpath(folder)))
            output_folder = os.path.join(os.path.dirname(folder), f"Outputs_{parent_folder}")
            os.makedirs(output_folder, exist_ok=True)
        
            # Preprocess images before adding to Metashape
            preprocessed_folder = os.path.join(output_folder, "Preprocessed_Images")
            process_success = False

            # Check if preprocessed images already exist with matching count
            if os.path.exists(preprocessed_folder):
                preprocessed_images = find_files(preprocessed_folder, valid_exts)
                if len(preprocessed_images) == len(photos):
                    print(f"Preprocessed images already exist in {preprocessed_folder}. Skipping preprocessing.")
                    process_success = True
                    
            # Perform preprocessing if needed
            if not process_success:
                print(f"{'Reprocessing' if os.path.exists(preprocessed_folder) else 'Preprocessing'} images with lens and vignette corrections...")
                process_success = process_rgb_images(folder, preprocessed_folder)

            # Handle the result of preprocessing
            if process_success:
                # Use preprocessed images
                photos = find_files(preprocessed_folder, valid_exts)
                print(f"Using {len(photos)} preprocessed images.")
            else:
                print("Using original images due to preprocessing failure.")
            
            # Create and open a new Metashape document
            doc = Metashape.Document()
            project_path = os.path.join(output_folder, f"project_{parent_folder}.psx")
            doc.save(project_path)
            doc.open(project_path, read_only=False, ignore_lock=True)
            
            # Create a new chunk and add preprocessed photos
            chunk = doc.addChunk()
            chunk.addPhotos(filenames=photos, progress=progress_timer.update)
            print(f"{len(chunk.cameras)} images loaded from: {preprocessed_folder if process_success else folder}.")
            doc.save()

            # Rename photos to include parent directory
            for camera in chunk.cameras:
                try:
                    path = os.path.normpath(camera.photo.path)
                    parent_dir = os.path.basename(os.path.dirname(path))
                    camera.label = f"{parent_dir}/{os.path.basename(path)}"
                except:
                    pass
            print("Camera labels updated with parent directory names.")

            try:
                print("Analyzing image quality...")
                chunk.analyzeImages(progress=progress_timer.update)
                
                # Get quality estimates
                quality_values = {}
                for camera in chunk.cameras:
                    if camera.meta["Image/Quality"] is not None:
                        quality_values[camera] = float(camera.meta["Image/Quality"])
                
                # Filter out low-quality images
                if quality_values:
                    quality_threshold = 0.75 
                    low_quality_cameras = [camera for camera, quality in quality_values.items() 
                                        if quality < quality_threshold]
                    
                    if low_quality_cameras:
                        print(f"Disabling {len(low_quality_cameras)} low-quality images (quality < {quality_threshold})...")
                        for camera in low_quality_cameras:
                            camera.enabled = False
                        print(f"{len([c for c in chunk.cameras if c.enabled])} high-quality images retained.")
                    else:
                        print("All images passed quality check.")
                else:
                    print("Image quality analysis did not return results. Proceeding with all images.")
                doc.save()
            except Exception as e:
                print(f"Error during image quality analysis for folder {folder}: {e}")
                continue

            try:
                # Match photos
                print("Matching photos...")
                timer_match = time.time()
                chunk.matchPhotos(downscale=0,
                                keypoint_limit=120000, 
                                keypoint_limit_3d=240000, 
                                keypoint_limit_per_mpx=6000,
                                tiepoint_limit=0, # no limit
                                generic_preselection=True, 
                                reference_preselection=True,
                                filter_mask=False, 
                                mask_tiepoints=True, 
                                filter_stationary_points=False,
                                keep_keypoints=True,
                                reset_matches=False, 
                                guided_matching=True,
                                progress=progress_timer.update)  
                print(f"Photos matched in {round(time.time() - timer_match, 1)} seconds.")
                
                # Align cameras
                chunk.alignCameras(adaptive_fitting=True, min_image=2, progress=progress_timer.update)
                print(f"{len(chunk.cameras)} cameras aligned.")
                
                # Reset region to prevent point clipping
                reset_region(chunk)
                
                # Disable lens distortion parameters since images are already corrected
                print("Optimizing cameras for preprocessed images...")
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                    fit_k1=False, fit_k2=False, fit_k3=False, 
                                    fit_k4=False, fit_p1=False, fit_p2=False,  
                                    fit_b1=False, fit_b2=False,
                                    tiepoint_covariance=True,
                                    progress=progress_timer.update)
                print(f"{len(chunk.cameras)} cameras optimized")
                doc.save()
            except Exception as e:
                print(f"Error during camera alignment/matching/optimization for folder {folder}: {e}")
                continue

            try:
                # Adaptive filtering approach
                print("Beginning adaptive tie point filtering...")
                
                # Store initial point count
                points_before = len(chunk.tie_points.points)
                
                # Filter by reconstruction uncertainty
                ru_threshold = adaptive_filter_reconstruction_uncertainty(chunk, initial_threshold=15)
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,
                                    fit_k1=False, fit_k2=False, fit_k3=False,  
                                    fit_k4=False, fit_p1=False, fit_p2=False,
                                    fit_b1=False, fit_b2=False,
                                    tiepoint_covariance=True,
                                    progress=progress_timer.update)
                
                # Filter by projection accuracy
                pa_threshold = adaptive_filter_projection_accuracy(chunk, initial_threshold=2)
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,
                                    fit_k1=False, fit_k2=False, fit_k3=False,
                                    fit_k4=False, fit_p1=False, fit_p2=False,
                                    fit_b1=False, fit_b2=False,
                                    tiepoint_covariance=True,
                                    progress=progress_timer.update)
                
                # Filter by reprojection error
                re_threshold = adaptive_filter_reprojection_error(chunk, initial_threshold=0.3)
                
                # Final optimization with adaptive fitting after all filtering
                print("Performing final camera optimization with adaptive fitting...")
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                    fit_k1=False, fit_k2=False, fit_k3=False,
                                    fit_k4=False, fit_p1=False, fit_p2=False,  
                                    fit_b1=False, fit_b2=False,
                                    adaptive_fitting=True,
                                    tiepoint_covariance=True,
                                    progress=progress_timer.update)
                
                # Report filtering results
                points_after = len(chunk.tie_points.points)
                percent_removed = round((points_before - points_after) / points_before * 100, 1)
                print(f"Tie point filtering complete. Removed {points_before - points_after} points ({percent_removed}%).")
                print(f"Applied thresholds: RU={ru_threshold}, PA={pa_threshold}, RE={re_threshold}")
                
                # Reset region again after filtering
                reset_region(chunk)
                
                # Export camera positions
                camera_file = os.path.join(output_folder, f"{parent_folder}_camera_positions.txt")
                chunk.exportCameras(camera_file, format=Metashape.CamerasFormat.CamerasFormatOPK)
                print("Camera positions exported.")
                doc.save()
            except Exception as e:
                print(f"Error during tie point filtering for folder {folder}: {e}")
                continue
            
            try:
                # Build depth maps
                print("Building depth maps...")
                chunk.buildDepthMaps(downscale=1, 
                                     filter_mode=Metashape.FilterMode.MildFiltering, 
                                     reuse_depth=True,
                                     max_neighbors=24,
                                     progress=progress_timer.update)
                print("Depth maps finished building.")
                doc.save()
            except Exception as e:
                print(f"Error building depth maps for folder {folder}: {e}")
                continue
            
            try:
                has_transform = (chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation)
                if has_transform:
                    # Build point cloud from depth maps
                    print("Building dense point cloud...")
                    chunk.buildPointCloud(source_data=Metashape.DataSource.DepthMapsData, 
                                          point_colors=True, 
                                          point_confidence=True, 
                                          keep_depth=True,
                                          max_neighbors=100,
                                          progress=progress_timer.update) 
                    print("Point cloud finished building.")
                    
                    # Filter point cloud
                    chunk.point_cloud.setConfidenceFilter(0, 1)
                    chunk.point_cloud.cropSelectedPoints()
                    chunk.point_cloud.setConfidenceFilter(0, 255)
                    chunk.point_cloud.compactPoints()
                    print("Point cloud filtered by confidence.")              

                    # Classify ground points with parameters
                    chunk.point_cloud.classifyGroundPoints(
                        cell_size=3.0,      
                        max_angle=5.0,       
                        max_distance=0.2,    
                        max_terrain_slope=6.0,
                        progress=progress_timer.update,
                    )
                    print("Point cloud ground points classified.")

                    # Low vegetation (0.05-0.5m above ground)
                    chunk.point_cloud.classifyPoints(
                        target_class=Metashape.PointClass.LowVegetation,
                        source_class=[Metashape.PointClass.Created, Metashape.PointClass.Unclassified],
                        from_class=Metashape.PointClass.Ground,
                        distance_above=0.05,  # Minimum height
                        distance_below=0.5,    # Maximum height
                        progress=progress_timer.update
                    )
                    print("Low vegetation classified.")

                    # Medium vegetation (0.5-2m above ground)
                    chunk.point_cloud.classifyPoints(
                        target_class=Metashape.PointClass.MediumVegetation,
                        source_class=[Metashape.PointClass.Created, Metashape.PointClass.Unclassified],
                        from_class=Metashape.PointClass.Ground,
                        distance_above=0.5,   
                        distance_below=2.0,   
                        progress=progress_timer.update
                    )
                    print("Medium vegetation classified.")

                    # High vegetation (>2m above ground)
                    chunk.point_cloud.classifyPoints(
                        target_class=Metashape.PointClass.HighVegetation,
                        source_class=[Metashape.PointClass.Created, Metashape.PointClass.Unclassified],
                        from_class=Metashape.PointClass.Ground,
                        distance_above=2.0,
                        progress=progress_timer.update
                    )
                    print("High vegetation classified.")

                    print("Point classes with points after classification:")
                    for attr in dir(Metashape.PointClass):
                        if not attr.startswith('__') and attr not in ['values']:
                            try:
                                class_value = getattr(Metashape.PointClass, attr)
                                count = chunk.point_cloud.getPointCount(class_value)
                                print(f"  {attr} (class {class_value}): {count} points")
                            except Exception as e:
                                pass

                    # Export point cloud
                    pc_file = os.path.join(output_folder, f"{parent_folder}_point_cloud.las")
                    chunk.exportPointCloud(pc_file, source_data=Metashape.DataSource.PointCloudData, progress=progress_timer.update)
                    print("Point cloud exported.")
                    doc.save()
                    gc.collect()
                    
                    # Build DTM (ground points only) 
                    print("Building Digital Terrain Model (ground points only)...")
                    chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                                interpolation=Metashape.Interpolation.EnabledInterpolation,
                                classes=[Metashape.PointClass.Ground],
                                progress=progress_timer.update)  # Only use ground points
                    print("DTM (terrain model) finished building.")

                    # Export DTM
                    dtm_file = os.path.join(output_folder, f"{parent_folder}_dtm.tif")
                    compression = Metashape.ImageCompression()
                    compression.tiff_big = True
                    chunk.exportRaster(dtm_file, 
                                       source_data=Metashape.DataSource.ElevationData, 
                                       image_compression=compression,
                                       progress=progress_timer.update)  
                    print("DTM exported")
                    doc.save()
                    gc.collect()  # Memory management after large export

                    # reset point filters
                    chunk.point_cloud.resetFilters()

                    # Build DSM using non-ground classes 
                    non_ground_classes = []
                    for class_name in ['Unclassified', 'Created', 'LowVegetation', 'MediumVegetation', 'HighVegetation']:
                        class_value = getattr(Metashape.PointClass, class_name)
                        non_ground_classes.append(class_value)

                    print(f"Using these non-ground classes for DSM: {non_ground_classes}")
                    chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                                interpolation=Metashape.Interpolation.EnabledInterpolation,
                                classes=non_ground_classes,
                                progress=progress_timer.update)
                    print("DSM (excluding ground points) finished building.")
                    
                    # Export DSM
                    dsm_file = os.path.join(output_folder, f"{parent_folder}_dsm.tif")
                    chunk.exportRaster(dsm_file, 
                                       source_data=Metashape.DataSource.ElevationData,
                                       image_compression=compression,
                                       progress=progress_timer.update)
                    print("DSM exported")
                    doc.save()
                    gc.collect()
                    
                    # Build model from depth maps
                    print("Building 3D mesh from depth maps...")
                    gc.collect()  # Force collection before heavy processing
                    chunk.buildModel(source_data=Metashape.DataSource.DepthMapsData, 
                                    surface_type=Metashape.SurfaceType.HeightField,  
                                    interpolation=Metashape.Interpolation.EnabledInterpolation,
                                    face_count=Metashape.FaceCount.HighFaceCount,
                                    progress=progress_timer.update)
                    print("Mesh built.")
                    
                    # Apply mesh improvements
                    print("Optimizing mesh...")
                    chunk.decimateModel(face_count=len(chunk.model.faces) / 2)  # Reduce mesh complexity
                    print("Mesh decimated.")
                    chunk.smoothModel(25) 
                    print("Mesh smoothed.")
                    doc.save()
                    gc.collect()
                    
                    # Build orthomosaic with improved parameters
                    print("Building orthomosaic...")
                    chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ModelData, 
                                         blending_mode=Metashape.BlendingMode.MosaicBlending, 
                                         ghosting_filter=True,  
                                         fill_holes=True,
                                         cull_faces=True,
                                         refine_seamlines=True,
                                         progress=progress_timer.update)
                    print("Orthomosaic finished building.")
                    
                    # Export orthomosaic with tiling
                    ortho_file = os.path.join(output_folder, f"{parent_folder}_orthomosaic.tif")
                    chunk.exportRaster(ortho_file, 
                                      source_data=Metashape.DataSource.OrthomosaicData,
                                      image_compression=compression,
                                      progress=progress_timer.update)
                    print("Orthomosaic exported.")
                    doc.save()
                    gc.collect()
            except Exception as e:
                print(f"Error during DEM/Model/Orthomosaic building for folder {folder}: {e}")
                continue
            
            try:
                # Export report
                report_file = os.path.join(output_folder, f"{parent_folder}_report.pdf")
                chunk.exportReport(report_file)
                print("Report exported.")
                print(f"Processing finished for folder {folder}; results saved to {output_folder}.")

                # Properly close the document before moving to next folder
                doc.save()
                doc = None  # Release the document reference
                gc.collect()  
                print("Document closed and memory released.")
            except Exception as e:
                print(f"Error during export for folder {folder}: {e}")
                doc.save()
                doc = None
                gc.collect()
                continue
            
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

if __name__ == '__main__':
    gc.collect()
    main()