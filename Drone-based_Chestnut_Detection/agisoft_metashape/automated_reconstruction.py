import Metashape
import os
import sys
import ast
import datetime

RU = 20
PA = 6
RE = 0.8

def find_files(folder, valid_types):
    try:
        valid_types = [ext.lower() for ext in valid_types]
        return [os.path.join(folder, entry.name)
                for entry in os.scandir(folder)
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_types]
    except Exception as e:
        print(f"Error scanning folder {folder}: {e}")
        return []

# gradual point selection using reconstruction uncertainty
def grad_selection_RU(chunk, RU_thrsh, num_tries=4, pct_del=10, thrsh_incr=1):
    try:
        n = 0
        target_thrsh = 10
        init_thrsh = RU_thrsh
        points = chunk.tie_points.points
        points_start_num = len(points)
        f = Metashape.TiePoints.Filter()
        f.init(chunk, criterion=Metashape.TiePoints.Filter.ReconstructionUncertainty)
        f.selectPoints(init_thrsh)
        nselected = len([1 for point in points if point.valid and point.selected])
        pct_selected = (nselected / points_start_num) * 100
        if pct_selected <= 1:
            while True:
                print(f"Current RU threshold is {init_thrsh}. Adjusting downward by {thrsh_incr}.")
                init_thrsh -= thrsh_incr
                f.selectPoints(init_thrsh)
                nselected = len([1 for point in points if point.valid and point.selected])
                ps = (nselected / points_start_num) * 100
                if ps >= 1:
                    break
        elif pct_selected > pct_del:
            while True:
                if pct_selected > pct_del:
                    init_thrsh += thrsh_incr
                    f.selectPoints(init_thrsh)
                    nselected = len([1 for point in points if point.valid and point.selected])
                    ps = (nselected / points_start_num) * 100
                    if ps < pct_del:
                        break
        print(f"New adjusted RU threshold is {init_thrsh}. Beginning gradual selection.")
        n += 1
        print(f"Removing {nselected} tie points (RU) at a threshold of {init_thrsh}.")
        f.removePoints(init_thrsh)
        init_thrsh -= thrsh_incr
        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                              fit_k1=True, fit_k2=True, fit_k3=True,  
                              fit_k4=False, fit_p1=False, fit_p2=False,  
                              fit_b1=False, fit_b2=False)
        points = chunk.tie_points.points
        npoints = len(points)
        while True:
            n += 1
            if n > num_tries or init_thrsh <= target_thrsh or (100 * ((points_start_num - npoints) / points_start_num)) >= 30:
                break
            else:
                points = chunk.tie_points.points
                npoints = len(points)
                f.selectPoints(init_thrsh)
                nselected = len([1 for point in points if point.valid and point.selected])
                pct_selected = (nselected / npoints) * 100
                while True:
                    if pct_selected <= pct_del:
                        init_thrsh -= thrsh_incr / 5
                        f.selectPoints(init_thrsh)
                        nselected = len([1 for point in points if point.valid and point.selected])
                        pct_selected = (nselected / npoints) * 100
                    else:
                        break
                f.selectPoints(init_thrsh)
                nselected = len([1 for point in points if point.valid and point.selected])
                if nselected > 0:
                    print(f"Removing {nselected} tie points at a threshold of {init_thrsh}.")
                    f.removePoints(init_thrsh)
                    init_thrsh -= thrsh_incr
                    chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                          fit_k1=True, fit_k2=True, fit_k3=True,  
                                          fit_k4=False, fit_p1=False, fit_p2=False,  
                                          fit_b1=False, fit_b2=False)
                    points = chunk.tie_points.points
                    npoints = len(points)
    except Exception as e:
        print(f"Error during gradual selection (RU): {e}")

# gradual point selection using projection accuracy
def grad_selection_PA(chunk, PA_thrsh, num_tries=4, pct_del=10, thrsh_incr=1):
    try:
        n = 0
        target_thrsh = 2
        init_thrsh = PA_thrsh
        points = chunk.tie_points.points
        points_start_num = len(points)
        npoints = len(points)
        f = Metashape.TiePoints.Filter()
        f.init(chunk, criterion=Metashape.TiePoints.Filter.ProjectionAccuracy)
        f.selectPoints(init_thrsh)
        nselected = len([1 for point in points if point.valid and point.selected])
        pct_selected = (nselected / points_start_num) * 100
        if init_thrsh <= target_thrsh and pct_selected <= 50:
            print(f"Now removing {nselected} tie points (PA) at a threshold of {init_thrsh}.")
            f.removePoints(init_thrsh)
            chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                  fit_k1=True, fit_k2=True, fit_k3=True,  
                                  fit_k4=False, fit_p1=False, fit_p2=False,  
                                  fit_b1=False, fit_b2=False)
        else:
            while True:
                n += 1
                if n > num_tries or init_thrsh <= target_thrsh or (100 * ((points_start_num - npoints) / points_start_num)) >= 50:
                    break
                else:
                    points = chunk.tie_points.points
                    npoints = len(points)
                    f.selectPoints(init_thrsh)
                    nselected = len([1 for point in points if point.valid and point.selected])
                    pct_selected = (nselected / npoints) * 100
                    while True:
                        if pct_selected <= pct_del:
                            init_thrsh -= thrsh_incr / 5
                            f.selectPoints(init_thrsh)
                            nselected = len([1 for point in points if point.valid and point.selected])
                            pct_selected = (nselected / npoints) * 100
                        else:
                            break
                    f.selectPoints(init_thrsh)
                    nselected = len([1 for point in points if point.valid and point.selected])
                    if nselected > 0:
                        print(f"Removing {nselected} tie points (PA) at a threshold of {init_thrsh}.")
                        f.removePoints(init_thrsh)
                        init_thrsh -= thrsh_incr
                        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                              fit_k1=True, fit_k2=True, fit_k3=True,  
                                              fit_k4=False, fit_p1=False, fit_p2=False,  
                                              fit_b1=False, fit_b2=False)
                        points = chunk.tie_points.points
                        npoints = len(points)
    except Exception as e:
        print(f"Error during gradual selection (PA): {e}")


def grad_selection_RE(chunk, RE_thrsh, num_tries=10, pct_del=10, thrsh_incr=0.05):
    try:
        n = 0
        target_thrsh = 0.2
        init_thrsh = RE_thrsh
        points = chunk.tie_points.points
        points_start_num = len(points)
        npoints = len(points)
        f = Metashape.TiePoints.Filter()
        f.init(chunk, criterion=Metashape.TiePoints.Filter.ReprojectionError)
        f.selectPoints(init_thrsh)
        nselected = len([1 for point in points if point.valid and point.selected])
        pct_selected = (nselected / points_start_num) * 100
        if init_thrsh <= target_thrsh and pct_selected <= pct_del:
            print(f"Now removing {nselected} tie points (RE) at a threshold of {init_thrsh}.")
            f.removePoints(init_thrsh)
            chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                  fit_k1=True, fit_k2=True, fit_k3=True,  
                                  fit_k4=False, fit_p1=False, fit_p2=False,  
                                  fit_b1=False, fit_b2=False)
        else:
            while True:
                n += 1
                if n > num_tries or init_thrsh <= target_thrsh or (100 * ((points_start_num - npoints) / points_start_num)) >= pct_del:
                    break
                else:
                    points = chunk.tie_points.points
                    npoints = len(points)
                    f.selectPoints(init_thrsh)
                    nselected = len([1 for point in points if point.valid and point.selected])
                    pct_selected = (nselected / npoints) * 100
                    while True:
                        if pct_selected <= pct_del:
                            init_thrsh -= thrsh_incr
                            f.selectPoints(init_thrsh)
                            nselected = len([1 for point in points if point.valid and point.selected])
                            pct_selected = (nselected / npoints) * 100
                        else:
                            break
                    f.selectPoints(init_thrsh)
                    nselected = len([1 for point in points if point.valid and point.selected])
                    if nselected > 0:
                        print(f"Removing {nselected} tie points (RE) at a threshold of {init_thrsh}.")
                        f.removePoints(init_thrsh)
                        init_thrsh -= thrsh_incr
                        chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                              fit_k1=True, fit_k2=True, fit_k3=True,  
                                              fit_k4=False, fit_p1=False, fit_p2=False,  
                                              fit_b1=False, fit_b2=False)
                        points = chunk.tie_points.points
                        npoints = len(points)
    except Exception as e:
        print(f"Error during gradual selection (RE): {e}")


def main():
    # Parse argument (still required)
    if len(sys.argv) != 2:
        print("Usage: python automated_reconsutrction.py \"['<images_folder1>', '<images_folder2>', ...]\"")
        sys.exit(1)
    
    try:
        folder_paths = ast.literal_eval(sys.argv[1])
        if not isinstance(folder_paths, list):
            raise ValueError("Provided argument is not a list.")
    except Exception as e:
        print(f"Error parsing argument: {e}")
        sys.exit(1)
        
    folder_paths = [os.path.normpath(p) for p in folder_paths]
    valid_exts = [".JPG", ".JPEG", ".TIF", ".TIFF"]

    # Process each folder in the list independently
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
            
            # Set timestamp and output folder based on the parent folder of the current folder
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_folder = os.path.dirname(os.path.normpath(folder))
            output_folder = os.path.join(base_folder, f"Outputs_{current_time}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Create and open a new Metashape document
            doc = Metashape.Document()
            project_path = os.path.join(output_folder, f"{current_time}_project.psx")
            doc.save(project_path)
            doc.open(project_path, read_only=False, ignore_lock=True)
            
            # Create a new chunk and add photos
            chunk = doc.addChunk()
            chunk.addPhotos(photos)
            print(f"{len(chunk.cameras)} images loaded for folder {folder}.")
            doc.save()

            try:
                # calibrate reflectance using sun sensor only
                chunk.calibrateReflectance(use_reflectance_panels=False,
                                           use_sun_sensor=True)
                print("Reflectance calibrated.")
                doc.save()
            except Exception as e:
                print(f"Error calibrating reflectance for folder {folder}: {e}")
                continue
            
            try:
                # Align cameras, match photos, and optimize cameras
                chunk.alignCameras(adaptive_fitting=False, min_image=2)
                print(f"{len(chunk.cameras)} cameras aligned.")
                
                chunk.matchPhotos(downscale=1, keypoint_limit=120000, keypoint_limit_3d=300000, 
                                  keypoint_limit_per_mpx=2000, tiepoint_limit=80000,
                                  generic_preselection=True, reference_preselection=True,
                                  filter_mask=True, mask_tiepoints=True, filter_stationary_points=True,
                                  keep_keypoints=False, reset_matches=False)
                print(f"{len(chunk.cameras)} cameras matched.")
                doc.save()
                
                chunk.alignCameras(adaptive_fitting=False, min_image=2)
                print(f"{len(chunk.cameras)} cameras re-aligned.")
                
                chunk.optimizeCameras(fit_f=True, fit_cx=True, fit_cy=True,  
                                      fit_k1=True, fit_k2=True, fit_k3=True,  
                                      fit_k4=False, fit_p1=False, fit_p2=False,  
                                      fit_b1=False, fit_b2=False)
                print(f"{len(chunk.cameras)} cameras optimized.")
                doc.save()
            except Exception as e:
                print(f"Error during camera alignment/matching/optimization for folder {folder}: {e}")
                continue
            
            try:
                # Gradual tie point filtering
                RU_thrsh = RU
                # PA_thrsh = PA
                RE_thrsh = RE 
                
                print("Performing gradual selection by Reconstruction Uncertainty.")
                grad_selection_RU(chunk, RU_thrsh, pct_del=5)
                
                # # Sometimes PA filtering is too much for tree canopies. 
                # print(f"Performing gradual selection using Projection Accuracy with threshold {PA_thrsh}.")
                # grad_selection_PA(chunk, PA_thrsh, pct_del=5)
                
                print(f"Performing gradual selection using Reprojection Error with threshold {RE_thrsh}.")
                grad_selection_RE(chunk, RE_thrsh, pct_del=5)
                doc.save()
            except Exception as e:
                print(f"Error during tie point filtering for folder {folder}: {e}")
                continue
            
            try:
                # Build depth maps
                chunk.buildDepthMaps(downscale=1, 
                                     filter_mode=Metashape.FilterMode.MildFiltering, 
                                     reuse_depth=True)
                print("Depth maps finished building.")
                doc.save()
            except Exception as e:
                print(f"Error building depth maps for folder {folder}: {e}")
                continue
            
            try:
                has_transform = (chunk.transform.scale and chunk.transform.rotation and chunk.transform.translation)
                if has_transform:
                    # Build point cloud from depth maps
                    chunk.buildPointCloud(source_data=Metashape.DataSource.DepthMapsData, 
                                          point_colors=True, 
                                          point_confidence=True, 
                                          keep_depth=True)
                    print("Point cloud finished building.")
                    chunk.point_cloud.setConfidenceFilter(0, 1)
                    chunk.point_cloud.cropSelectedPoints()
                    chunk.point_cloud.setConfidenceFilter(0, 255)
                    chunk.point_cloud.compactPoints()
                    print("Point cloud filtered.")
                    chunk.point_cloud.classifyGroundPoints(cell_size=0.25)
                    print("Point cloud ground points classified.")
                    doc.save()
                    
                    # Build DEM, Model, and Orthomosaic
                    chunk.buildDem(source_data=Metashape.DataSource.PointCloudData,
                                   interpolation=Metashape.Interpolation.EnabledInterpolation)
                    print("DEM finished building.")
                    doc.save()
                    
                    chunk.buildModel(source_data=Metashape.DataSource.DepthMapsData, 
                                     surface_type=Metashape.SurfaceType.Arbitrary, 
                                     interpolation=Metashape.Interpolation.EnabledInterpolation,
                                     face_count=Metashape.FaceCount.HighFaceCount)
                    print("Mesh finished building.")
                    doc.save()
                    
                    chunk.buildOrthomosaic(surface_data=Metashape.DataSource.ModelData, 
                                           blending_mode=Metashape.BlendingMode.MosaicBlending,
                                           ghosting_filter=True,
                                           fill_holes=True,
                                           cull_faces=False,
                                           refine_seamlines=True)
                    print("Orthomosaic finished building.")
                    doc.save()
            except Exception as e:
                print(f"Error during DEM/Model/Orthomosaic building for folder {folder}: {e}")
                continue
            
            try:
                # Export results
                if chunk.point_cloud:
                    pc_file = os.path.join(output_folder, f"{current_time}_point_cloud.las")
                    chunk.exportPointCloud(pc_file, source_data=Metashape.DataSource.PointCloudData)
                    print("Point cloud exported.")
                
                if chunk.elevation:
                    dem_file = os.path.join(output_folder, f"{current_time}_dem.tif")
                    chunk.exportRaster(dem_file, source_data=Metashape.DataSource.ElevationData)
                    print("DEM exported.")
                
                if chunk.orthomosaic:
                    ortho_file = os.path.join(output_folder, f"{current_time}_orthomosaic.tif")
                    chunk.exportRaster(ortho_file, source_data=Metashape.DataSource.OrthomosaicData)
                    print("Orthomosaic exported.")
                
                report_file = os.path.join(output_folder, f"{current_time}_report.pdf")
                chunk.exportReport(report_file)
                print("Report exported.")
    
                print(f"Processing finished for folder {folder}; results saved to {output_folder}.")
            except Exception as e:
                print(f"Error during export for folder {folder}: {e}")
                continue
            
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue

if __name__ == '__main__':
    main()