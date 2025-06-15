import torch
import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import SmallUNetSGM

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the input size used during training
    transforms.ToTensor()
])

def preprocess(image_path, mask_path):
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # Grayscale for mask
    original_size = image.size  # Keep original size
    image = transform(image)
    mask = transform(mask)
    return image, mask, original_size

torch_model = SmallUNetSGM(in_channels=3, out_channels=1)
torch_model.load_state_dict(torch.load('best_smallunet_lane_detection_20.pth', map_location=torch.device('cpu'))['model_state_dict'])

image_path = './Dataset_small/100k_images_val_small_data/bdd100k/images/100k/val/b1e0c01d-dd9e6e2f.jpg'
mask_path = './Dataset_small/bdd100k_lane_labels_trainval_small_data/bdd100k/labels/lane/masks/val/b1e0c01d-dd9e6e2f.png'
image, mask, original_size = preprocess(image_path, mask_path)
image = image.unsqueeze(0)  # Add batch dimension, no need to move to GPU
example_inputs = (image,)

torch.onnx.export(
    torch_model,
    example_inputs,
    "smallunet_model_test.onnx",
    opset_version=11,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
)