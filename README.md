# Image Enhancement Studio

A computer vision project that explores image reconstruction, super-resolution, and image quality evaluation using PyTorch and Real-ESRGAN.

## Overview

Image Enhancement Studio provides an interface for enhancing photographs and illustrations, comparing original and processed images, and downloading enhanced results.

The project combines a custom convolutional neural network experiment with a pretrained Real-ESRGAN model.

## Features

- Image upload with support for JPG, JPEG, PNG, and WEBP.
- Image enhancement using Real-ESRGAN.
- Before-and-after image comparison.
- Output resolution information.
- Downloadable enhanced images.
- An experimental PyTorch CNN for learning and comparing reconstruction methods.
- Evaluation using Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) in the accompanying notebook.

## Technologies

- Python
- PyTorch and Torchvision
- Streamlit
- NumPy
- Pillow
- OpenCV
- scikit-image
- Real-ESRGAN

## Project Structure

```text
ai-image-enhancement/
├── app.py
├── ai_enhancement_project.ipynb
├── image_enhancement_cnn.pth
├── requirements.txt
├── .gitignore
└── README.md
```

The Real-ESRGAN executable and model files are downloaded separately and are not included in this repository.

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ai-image-enhancement
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your repository.

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up Real-ESRGAN

Download the Windows release from the official repository:

https://github.com/xinntao/Real-ESRGAN/releases

Extract the release into the project directory. Ensure `app.py` can locate the `realesrgan-ncnn-vulkan.exe` executable and its accompanying `models` folder.

The application currently uses the `realesrgan-x4plus` model for general-purpose image enhancement.

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## Experimental CNN

A custom CNN was implemented and trained using CIFAR-10 image pairs.

The experiment reduces 32 × 32 images to 8 × 8 pixels, enlarges them with bicubic interpolation, and trains the CNN to reconstruct the original images. Reconstruction quality is evaluated using PSNR and SSIM.

The CNN is intended as a learning experiment. It was trained on small images and is not a general-purpose photograph restoration model.

## Limitations

- Enhancement quality depends on the input image and the selected pretrained model.
- Super-resolution may introduce plausible details that were not present in the original image.
- The experimental CNN is designed for small CIFAR-10 images and may not improve ordinary photographs.
- Real-ESRGAN must be installed separately before its enhancement functionality can be used.

## Acknowledgements

This project uses the open-source Real-ESRGAN implementation:

https://github.com/xinntao/Real-ESRGAN

Please refer to the original repository for licensing, attribution, and usage terms.

## Author

Developed as a computer vision and deep learning portfolio project.