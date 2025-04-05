import fathomnet.api.images as fnet_images
import requests
import json
import os

concepts = ["Acanthascinae", "Lampocteis cruentiventer", "Bathochordaeus"] 
limit = 200

base_dir = "fathomnet_data"
os.makedirs(base_dir, exist_ok=True)

for concept_name in concepts:
    concept_dir = os.path.join(base_dir, concept_name.replace(" ", "_"))
    images_dir = os.path.join(concept_dir, "images")
    os.makedirs(images_dir, exist_ok=True)  

    images = fnet_images.find_by_concept(concept=concept_name)[:limit]

    formatted_data = []
    for idx, img in enumerate(images):
        img_url = img.url 
        img_filename = f"{concept_name.replace(' ', '_')}_{idx}.jpg"  
        img_path = os.path.join(images_dir, img_filename)

        if img_url:
            try:
                img_data = requests.get(img_url).content
                with open(img_path, "wb") as f:
                    f.write(img_data)
                print(f"Downloaded: {img_filename} for {concept_name}")
            except Exception as e:
                print(f"Failed to download {img_filename}: {e}")

        formatted_data.append({
            "uuid": img.uuid,
            "url": img.url,
            "local_path": img_path, 
            "contributorsEmail": img.contributorsEmail,
            "width": img.width,
            "height": img.height,
            "imagingType": img.imagingType,
            "salinity": img.salinity,
            "temperatureCelsius": img.temperatureCelsius,
            "oxygenMlL": img.oxygenMlL,
            "latitude": img.latitude,
            "longitude": img.longitude,
            "sha256": img.sha256,
            "createdTimestamp": img.createdTimestamp,
            "lastValidation": img.lastValidation,
            "valid": img.valid,
            "lastUpdatedTimestamp": img.lastUpdatedTimestamp,
            "tags": [
                {
                    "uuid": tag.uuid,
                    "key": tag.key,
                    "value": tag.value,
                    "mediaType": tag.mediaType
                } for tag in img.tags
            ] if img.tags else [],
            "boundingBoxes": [
                {
                    "uuid": bbox.uuid,
                    "userDefinedKey": bbox.userDefinedKey,
                    "concept": bbox.concept,
                    "height": bbox.height,
                    "observer": bbox.observer,
                    "width": bbox.width,
                    "x": bbox.x,
                    "y": bbox.y,
                    "rejected": bbox.rejected,
                    "verified": bbox.verified,
                    "lastUpdatedTimestamp": bbox.lastUpdatedTimestamp,
                    "createdTimestamp": bbox.createdTimestamp
                } for bbox in img.boundingBoxes
            ] if img.boundingBoxes else []
        })

    json_filename = os.path.join(concept_dir, f"{concept_name.replace(' ', '_')}_images.json")
    with open(json_filename, "w") as json_file:
        json.dump(formatted_data, json_file, indent=4)

    print(f"JSON file saved: {json_filename}")

print("All downloads complete!")
