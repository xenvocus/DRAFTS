import torch
from BinaryClass.binary_model import BinaryNet

# --- Parameters ---
base_model = 'resnet18'
model_path = './class_resnet18.pth'
onnx_model_path = './class_resnet18.onnx'
batch_size = 16 # Example batch size, can be dynamic
height, width = 512, 512

# --- Load PyTorch Model ---
model = BinaryNet(base_model, num_classes=2)
model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
model.eval()

# --- Export to ONNX ---
# Create a dummy input with the correct shape
dummy_input = torch.randn(batch_size, 1, height, width)

# Set dynamic axes to allow for variable batch size
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

torch.onnx.export(model,
                  dummy_input,
                  onnx_model_path,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes=dynamic_axes)

print(f"Model has been converted to ONNX format and saved at {onnx_model_path}")