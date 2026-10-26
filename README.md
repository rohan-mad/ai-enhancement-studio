# Image Enhancement Studio

A computer vision project exploring image enhancement, super-resolution, and image reconstruction using Python, PyTorch, and Real-ESRGAN.

The application allows users to upload photographs and illustrations, enhance images, compare original and processed results, and download the enhanced output. It also includes a custom convolutional neural network (CNN) experiment developed and trained using PyTorch.

## Live Demo

[Open Image Enhancement Studio](https://ai-enhancement-studio-k35czvpq6zxfpmnbtwojdg.streamlit.app/)

## Application Preview

### Main Interface

![Image Enhancement Studio main interface](screenshots/intro.png)

### Before and After Enhancement

![Original and enhanced image comparison](screenshots/result.png)

## Features

- Upload images in JPG, JPEG, PNG, and WEBP formats.
- Enhance full-resolution images using a pretrained Real-ESRGAN model.
- Compare original and enhanced images side by side.
- View original and output image resolutions.
- Download enhanced images as PNG files.
- Experiment with a custom PyTorch CNN.
- Compare bicubic interpolation and CNN reconstruction using PSNR and SSIM.
- Run the application locally or use the hosted version.

## Technologies Used

- **Python** — application development
- **PyTorch and Torchvision** — CNN implementation and training
- **Real-ESRGAN** — pretrained image enhancement and super-resolution
- **Streamlit** — interactive web interface
- **NumPy and Pillow** — image handling and array operations
- **OpenCV** — image processing
- **Matplotlib** — image visualization
- **scikit-image** — image quality evaluation

## How It Works

The project combines two approaches to image enhancement.

### 1. Real-ESRGAN

The application integrates a pretrained Real-ESRGAN model for enhancing full-resolution photographs and illustrations. The current implementation uses `realesrgan-x4plus`, which produces an output with four times the original width and height.

A separate cloud-compatible application is provided in `cloud/app.py`. It downloads the Linux Real-ESRGAN executable and required model files when needed, while the main `app.py` uses the Windows executable.

Real-ESRGAN is a pretrained model integrated into this project; it was not trained from scratch as part of this work.

### 2. Custom CNN Experiment

A convolutional neural network was implemented and trained using the CIFAR-10 dataset.

The reconstruction pipeline is:

1. Start with an original 32 × 32 image.
2. Reduce it to 8 × 8 pixels.
3. Enlarge it to 32 × 32 using bicubic interpolation.
4. Train the CNN to predict corrections to the bicubic reconstruction.
5. Evaluate reconstruction quality using PSNR and SSIM.

The model uses convolutional layers, ReLU activation functions, an L1 loss function, and the Adam optimizer. Residual learning allows the model to predict corrections to the bicubic input rather than reconstructing the entire image from scratch.

This CNN experiment demonstrates the training and evaluation process. Because the model was trained on small CIFAR-10 images, it is not intended to be a general-purpose photograph restoration system.

## Evaluation

The project uses two image quality metrics.

- **Peak Signal-to-Noise Ratio (PSNR):** Measures pixel-level reconstruction error. Higher values generally indicate lower error.
- **Structural Similarity Index (SSIM):** Measures structural similarity between a reconstructed image and its reference. Scores closer to 1 generally indicate greater similarity.

The notebook compares bicubic interpolation with the CNN on held-out CIFAR-10 test images. Results depend on the dataset, preprocessing, training configuration, and model.

The experiments demonstrated that a CNN does not automatically outperform traditional interpolation. Visual inspection and quantitative evaluation were both important when investigating model behaviour and unwanted colour changes.

## Project Structure

```text
ai-enhancement-studio/
├── app.py
├── cloud/
│   ├── app.py
│   └── requirements.txt
├── ai_enhancement_project.ipynb
├── image_enhancement_cnn.pth
├── requirements.txt
├── packages.txt
├── .gitignore
├── README.md
└── screenshots/
    ├── intro.png
    └── result.png
```

The Real-ESRGAN executable and its model files are not included in the repository. The cloud version downloads its Linux executable and model files when required.

## Getting Started

### Prerequisites

- Python 3.12 recommended for the cloud configuration
- Git
- The appropriate Real-ESRGAN executable and model files for local Windows use

### 1. Clone the repository

```bash
git clone https://github.com/rohan-mad/ai-enhancement-studio.git
cd ai-enhancement-studio
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up Real-ESRGAN for local use

Download the appropriate Windows release from the official repository:

https://github.com/xinntao/Real-ESRGAN/releases

Extract the executable and its accompanying `models` folder into the project directory. The local application expects the extracted folder to be named either:

- `realesrgan-ncnn-vulkan-20220424-windows`
- `realesrgan`

Ensure that `realesrgan-ncnn-vulkan.exe` and the model files are present.

### 5. Run the local application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### 6. Run the CNN experiment

Open `ai_enhancement_project.ipynb` in Jupyter Notebook or VS Code and run the notebook cells to explore data preparation, image degradation, bicubic interpolation, CNN training, visualization, and evaluation.

The notebook downloads CIFAR-10 when needed.

## Cloud Deployment

The cloud application is located at `cloud/app.py` and uses `cloud/requirements.txt`. The repository also includes `packages.txt` for Linux system dependencies.

The application can be deployed through Streamlit Community Cloud by connecting the GitHub repository and selecting:

- **Branch:** `main`
- **Main file path:** `cloud/app.py`

The cloud version downloads the Linux Real-ESRGAN executable and model files at runtime. Processing speed and reliability depend on the hosting environment and its available resources.

## Limitations

- The custom CNN was trained on small CIFAR-10 images and may not improve ordinary photographs.
- Super-resolution can generate plausible details that were not present in the original image.
- Enhancement quality varies with the input image and model.
- Real-ESRGAN is a pretrained model integrated into the application, not a model trained from scratch for this project.
- Cloud processing may be slower than local processing on a computer with a dedicated GPU.
- Large images may require additional processing time or exceed hosting resource limits.

## Acknowledgements

This project uses the open-source Real-ESRGAN implementation:

https://github.com/xinntao/Real-ESRGAN

Refer to the original repository and the specific model's terms for licensing, attribution, and usage requirements.

## Author

Developed as a computer vision and deep learning portfolio project.