## https://cran.r-project.org/web/packages/ForestTools/vignettes/treetop_analysis.html

install.packages("ForestTools")

# Attach the 'ForestTools' and 'terra' libraries
library(ForestTools)
library(terra)
library(sf)

# Load normalized canopy height model
chm <- terra::rast("S:/Zack/Imagery/Hazelnut/WPP_Farm_LiDAR/ArcGIS_Pro_WPP_Farm_LiDAR/WPP_Farm_BlockB_SfM_CHM_Clip.tif")

# Remove plot margins (optional)
par(mar=rep(0.5,4))

# Plot CHM (extra optional arguments remove labels and tick marks from the plot)
plot(chm, xlab="", ylab="", xaxt='n', yaxt='n')

# Function for defining dynamic window size
#lin <- function(x){x * 0.06 + 0.5}

# Detect treetops > minHeight
#ttops <- vwf(chm, winFun = lin, minHeight = 1.5)

# Write ttops to shapefile
#sf::st_write(ttops, "S:/Zack/Imagery/Hazelnut/WPP_Farm_LiDAR/ArcGIS_Pro_WPP_Farm_LiDAR/WPP_Farm_W_treetops.shp", append=FALSE)

# load ttops from shapefile
ttops <- sf::st_read("S:/Zack/Imagery/Hazelnut/WPP_Farm_LiDAR/ArcGIS_Pro_WPP_Farm_LiDAR/wpp_farm_treetops_blockB_SfM.shp")

# make sure ttops is same crs as chm
ttops <- st_transform(ttops, st_crs(chm))

# Plot CHM
plot(chm, xlab = "", ylab = "", xaxt='n', yaxt = 'n')

# Add dominant treetops to the plot
plot(ttops$geometry, col = "red", pch = 20, cex = 0.5, add = TRUE)

# Get the mean treetop height (m)
# mean(ttops$height)

# Create crown map
crowns_ras <- mcws(treetops = ttops, CHM = chm, minHeight = 0.3)

# Plot crowns
plot(crowns_ras, col = sample(rainbow(50), nrow(unique(chm)), replace = TRUE), legend = FALSE, xlab = "", ylab = "", xaxt='n', yaxt = 'n')

# Create polygon crown map
crowns_poly <- mcws(treetops = ttops, CHM = chm, format = "polygons", minHeight = 0.3)

# Plot CHM
plot(chm, xlab = "", ylab = "", xaxt='n', yaxt = 'n')

# Add crown outlines to the plot
plot(crowns_poly$geometry, border = "blue", lwd = 0.5, add = TRUE)

# Compute area and diameter
crowns_poly[["area"]] <- st_area(crowns_poly)
crowns_poly[["diameter"]] <- sqrt(crowns_poly[["area"]]/ pi) * 2

# Mean crown diameter
mean(crowns_poly$diameter)

# Write polygon crowns to shapefile
sf::st_write(crowns_poly, "S:/Zack/Imagery/Hazelnut/WPP_Farm_LiDAR/ArcGIS_Pro_WPP_Farm_LiDAR/WPP_Farm_blockB_SfM_canopies.shp", append=FALSE)
