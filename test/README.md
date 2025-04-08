
## Research Proposal

### 1. Training Tasks
* Road segmentation
* Road surface classification (asphalt, concrete, paving stones, sett, unpaved)


### 2. Dataset Fusion
* [StreetSurfaceVis](https://zenodo.org/records/11449977): detailed road surface types and qualities s with Mapillary contributor names and image IDs.  
* Mapillary Vistas Dataset: provides various road types labels, but for surface classification need additional labellings

### 3. Model Desigin
* Feature extraction (ResNet, EfficientNet)
* Multi-Task model (combine segmentation and classification for sharing features)


### 4. Data Preprocessing
#### 4.1 Extract the road region from Mapillary Vistas Dataset
* create a mask, including labels from the origin dataset

        "construction--flat--road",
        "construction--flat--sidewalk",               
        "construction--flat--crosswalk-plain",  
        "construction--flat--bike-lane",
        "construction--flat--service-lane",
        "marking--discrete--crosswalk-zebra",  
        "marking--discrete--arrow--left",    
        "marking--discrete--arrow--right",   
        "marking--discrete--arrow--straight",  
        "marking--discrete--symbol--bicycle",
        "marking--discrete--stop-line",  
        "marking--continuous--dashed",  
        "marking--continuous--solid"
        
<img src="./Road_test_plot.png" alt="Road Mask Example" width="500"/>

<img src="./image_overview.png" alt="Road Mask Example" width="500"/>

#### 4.2 Pre-training model for surface classification based on StreetSurfaceVis

project/
├── data/
│   ├── raw/                  # Raw dataset (linked to /test/StreetSurfaceVis_1024)
│   └── processed/            # Processed data (e.g., train/val/test splits)
├── code/
│   ├── src/
│   │   ├── datasets/
│   │   │   └── street_surface_loader.py      # dataset loading and preprocessing
│   │   ├── models/
│   │   │   ├── efficient_net_classifier.py   # defines EfficientNet-based classifier
│   │   │   └── model_utils.py                # contains utility functions for model-based tasks
│   │   ├── trainers/
│   │   │   ├── train_classifier.py           # classifys train and validation datasets
│   │   │   └── metrics.py                    # computes evaluation metrics
│   │   ├── utils/
│   │   │   ├── config.py                     # loads and saves configuration files
│   │   │   ├── logger.py
│   │   │   └── utils.py                      # miscellaneous utility functions
│   │   └── visualization/                   
│   │       ├── visualize_predictions.py
│   │       ├── plot_loss.py
│   │       └── plot_metrics.py
│   ├── notebooks/
│   ├── scripts/
│   │   └── train.py
│   ├── config/
│   │   └── street_surface.yaml
│   ├── logs/
│   ├── models/
│   └── results/
├── jobs/
│   ├── setup_job.sh
│   └── train_job.sh
└── environment.yml


#### 4.3 Fine-tuning model on Mapillary Vistas Dataset (transfer learning)

#### 4.4 Multi-task learning- segmentation & Classification



### References
[1] Kapp, Alexandra, Edith Hoffmann, Esther Weigmann, and Helena Mihaljević. “StreetSurfaceVis: A Dataset of Crowdsourced Street-Level Imagery Annotated by Road Surface Type and Quality.” Scientific Data 12, no. 1 (January 16, 2025): 92. https://doi.org/10.1038/s41597-024-04295-9.



## Paper Timetable
<img src="./Paper_structure.png" alt="Paper_structure.png" width="800"/>




