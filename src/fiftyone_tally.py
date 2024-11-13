# fiftyone_tally.py is an exploratory analysis script using python's fifty-one package to 
# count the number of annotations and explore similarity of images in an interactive browser window.

import fiftyone as fo

name = "pcb-x"

dataset_dir = "pcb-vision"

dataset_type = fo.types.COCODetectionDataset

dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=dataset_type,
    name=name,
)

# parameters expect the data to be kept in an "images" directory
data_path = "pcb-vision/data"

labels_path = "data/result.json"

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
)

session = fo.launch_app(dataset)