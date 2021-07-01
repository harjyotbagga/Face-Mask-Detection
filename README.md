# Face Mask Detection
[![Issues][issues-shield]][issues-url]  

Face Mask Detection using Deep Learning.

## Frameworks & Tools used:
- Deep Learning Framework: PyTorch
- Face Detection Model: Single Shot Detector (SSD) framework using a ResNet-10 as Base Network (Shipped with OpenCV)
- Mask Detection: transfer learned classifier based on MobileNetV2

## Directory Structure
```
    .
    ├── dataset                 # Dataset to train the model (along with proper augmentation)
    ├── examples                # Real-life example images to test the model on
    ├── OG Dataset              # Orignal dataset to perform augmentations on
    ├── outputs                 # Output facial images produced after image detection
    ├── Training Models         # Weights of the trained models
    ├── image_augmentation.py   # Script to augment the orignal dataset & produce more examples for training
    ├── image_detection.ipynb   # Face Mask detection from Image
    ├── torch_transforms.ipynb  # Output of pre-training transformations of images
    ├── training.ipynb          # Model Training
    └── video-detection.py      # Face Mask detection from Live Camera Feed
```

## Installation & Excecution Steps
1. Clone this repository
    ```sh
    git clone https://github.com/harjyotbagga/face-mask-detection.git
    ```
2. Install the necessary requirements. (Mainly PyTorch, opencv-python and other basic data science tools)
3. Run ```video-detection.py```
    ```
    cd face-mask-detection
    python video-detection.py
    ```

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<div align="center">

### Made with ❤️
</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-shield]: https://img.shields.io/github/issues/harjyotbagga/face-mask-detection.svg?style=flat-square
[issues-url]: https://github.com/harjyotbagga/face-mask-detection/issues