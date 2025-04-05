import json
import csv
import os

base_dir = "fathomnet_data"
csv_filename = "fathomnet_images.csv"

with open(csv_filename, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'path', 'width', 'height', 'boundingBoxes_height',
        'boundingBoxes_width', 'boundingBoxes_x', 'boundingBoxes_y', 'name'
    ])

    for concept_folder in os.listdir(base_dir):
        concept_path = os.path.join(base_dir, concept_folder)

        if os.path.isdir(concept_path):
            json_filename = f"{concept_folder}_images.json"
            json_path = os.path.join(concept_path, json_filename)

            if os.path.exists(json_path):
                with open(json_path, "r") as file:
                    data = json.load(file)

                for idx, img in enumerate(data):
                    local_img_path = os.path.join(concept_path, "images", f"{concept_folder}_{idx}.jpg")

                    bounding_boxes = img.get("boundingBoxes", [])
                    bbox = bounding_boxes[0] if bounding_boxes else {}

                    csv_writer.writerow([
                        local_img_path,
                        img.get("width", "N/A"),
                        img.get("height", "N/A"),
                        bbox.get("height", "N/A"),
                        bbox.get("width", "N/A"),
                        bbox.get("x", "N/A"),
                        bbox.get("y", "N/A"),
                        concept_folder
                    ])

print(f"CSV file saved: {csv_filename}")
