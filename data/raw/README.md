# Real imagery

Put Chandrayaan-2, LRO, or SELENE products here after you have the right to use them.
This folder is not shipped with mission data.

Recommended path:

1. Download a subset from the data provider.
2. Convert PDS images to GeoTIFF with GDAL or ISIS if needed.
3. Write a sidecar JSON next to the file using `metadata_template.json`.
4. Leave unknown fields out. Do not invent sun angles, GSD, or product IDs.

Providers:

- Chandrayaan-2 imaging products: ISRO ISSDC / PRADAN (registration required). Browse: https://chmapbrowse.issdc.gov.in/
- LRO NAC: NASA PDS / LROC, http://lroc.sese.asu.edu/
- SELENE (Kaguya) Terrain Camera: JAXA DARTS

A raw `.img` file can be read only if the sidecar gives `width`, `height`, `dtype`, and `offset_bytes`. A GeoTIFF is the easier input. `rasterio` is optional; without it, pixel values still load and GeoTIFF CRS tags may be incomplete.

Nothing in this folder is loaded automatically.
