import pandas as pd
import requests
import os
import math
import time
from tqdm import tqdm

def get_tile_coords(lat, lon, zoom):
    """
    Converts Latitude and Longitude to Slippy Map Tile Coordinates (X, Y).
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_property_images(csv_path, output_dir, zoom=18):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Starting download for {len(df)} properties...")

    # Iterate through the rows
    # Note: Using a slice [:50] here for your first test run is recommended
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prop_id = row['id']
        lat = row['lat']
        lon = row['long']

        # Define filename
        filename = os.path.join(output_dir, f"{prop_id}.jpg")

        # Skip if already downloaded
        if os.path.exists(filename):
            continue

        try:
            # 1. Get Tile Coordinates
            xtile, ytile = get_tile_coords(lat, lon, zoom)

            # 2. Esri World Imagery URL
            url = f"https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{zoom}/{ytile}/{xtile}"

            # 3. Request and Save
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"\nFailed to download ID {prop_id}: Status {response.status_code}")

            # Respect the server with a tiny pause
            time.sleep(0.1)

        except Exception as e:
            print(f"\nError processing ID {prop_id}: {e}")
