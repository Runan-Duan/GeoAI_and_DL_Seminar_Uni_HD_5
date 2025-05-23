# Benchmarking Deep Learning Models for Road Surface Classification on StreetSurfaceVis

## Abstract
Accurate road surface classification is essential for enhancing travel time estimation in routing engines and supporting autonomous navigation systems. Recent advances in deep learning have enabled significant progress in visual surface recognition, yet comparative assessments of state-of-the-art architectures on standardized datasets remain limited. In this study, we conduct a systematic evaluation of three high-performing convolutional neural network architectures - ResNet-50, ConvNeXt-Small, and EfficientNet-B4 - on the StreetSurfaceVis benchmark dataset, which comprises diverse road surface types under varying lighting and environmental conditions. Despite overall high classification accuracy across models, performance disparities are observed across surface categories. ConvNeXt-Small achieves the highest class-wise performance, with an F1 score of 0.97 for paving stones, whereas concrete surfaces are consistently misclassified, yielding a maximum F1 score of 0.87. All models exhibit substantial confusion between asphalt and concrete, indicating limitations in discriminating visually similar textures using RGB data alone. These findings suggest that fine-grained material classification may benefit from model architectures incorporating attention mechanisms, texture-aware encoding, or multi-modal input such as multi-spectral images. While architectural differences have minimal impact on average performance, our results emphasize the importance of addressing class-level ambiguity through targeted model design and data-driven strategies. This study provides a rigorous baseline for future research in road surface understanding and contributes to the development of more robust vision-based infrastructure analysis systems.


## Directory Structure
* data: contains dataset description and labeling details
* log: provides experiment recordings
* output: includes model evaluation details (confusion matrix, accuracy, and visualization of samples)
* src: consists of data loader, trainer, models, visualizer for data preprocessing, training, model selection and visualization.
  
## License
This project is licensed under the MIT License.
