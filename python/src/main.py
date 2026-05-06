import os
import torch
import cv2
import numpy as np
import pandas as pd
from onnx import numpy_helper
from ultralytics import YOLO
import openvino.runtime as ov



FP_MODEL = "../models/yolo26n.pt"
SAMPLE_IMAGE = "../assets/bus.jpg"
INT8_MODEL_PATH = "../models/yolo26n_int8.pt"
LOG_DIR = "../logs/int8_extraction"


def Yolo_Detect_FP(model):
    model = YOLO(model)
    # Add save=True to generate the annotated image
    results = model.predict(SAMPLE_IMAGE, device="cuda", save=True)

    # This will print where the image was saved (usually runs/detect/predict/)
    print(f"Result saved to: {results[0].save_dir}")

    results = model.predict(SAMPLE_IMAGE, device="cuda")

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        
        for box in boxes:
            # Get coordinates in x1, y1, x2, y2 format
            coords = box.xyxy[0].tolist()
            # Get confidence score
            conf = box.conf[0].item()
            # Get class ID
            cls = int(box.cls[0].item())
            # Get class name
            name = result.names[cls]
            
            print(f"Object: {name} | Confidence: {conf:.2f} | BBox: {coords}")


def get_yolo26_details(model_path, log_dir="../logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, "model_structure.log")
    model = YOLO(model_path)
    inner_model = model.model
    
    with open(log_filepath, "w") as f:
        header = f"{'Idx':<5} | {'Type':<15} | {'Params':<10} | {'In-Ch':<12} | {'Out-Ch':<12}"
        separator = "-" * 70
        print(header)
        print(separator)
        f.write(header + "\n" + separator + "\n")

        total_params = 0
        for i, m in enumerate(inner_model.model):
            params = sum(p.numel() for p in m.parameters())
            total_params += params
            
            in_ch, out_ch = "N/A", "N/A"
            
            # 1. Standard Convolutions
            if hasattr(m, 'conv'):
                in_ch = getattr(m.conv, 'in_channels', "N/A")
                out_ch = getattr(m.conv, 'out_channels', "N/A")
            
            # 2. C3k2 / C2f Blocks
            elif hasattr(m, 'cv1'):
                in_ch = getattr(m.cv1.conv, 'in_channels', "N/A")
                if hasattr(m, 'cv2'):
                    out_ch = getattr(m.cv2.conv, 'out_channels', "N/A")
            
            # 3. Handle the Detect Head (where it was crashing)
            elif "Detect" in m.__class__.__name__:
                # Try common names for input channels in Ultralytics heads
                in_ch = getattr(m, 'ch', "Multi-Scale") 
                out_ch = getattr(m, 'nc', "N/A") # nc = number of classes

            line = f"{i:<5} | {m.__class__.__name__:<15} | {params:<10,} | {str(in_ch):<12} | {str(out_ch):<12}"
            print(line)
            f.write(line + "\n")

        footer = f"{separator}\nTotal Parameters: {total_params:,}"
        print(footer)
        f.write(footer + "\n")

    print(f"\n✅ Log successfully saved to: {log_filepath}")




def extract_layer_weights(model_path, log_dir="../logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, "layer_weights.log")
    model = YOLO(model_path)
    inner_model = model.model
    
    with open(log_filepath, "w") as f:
        header = f"{'Idx':<5} | {'Type':<15} | {'Weight Shape':<20} | {'Bias Shape':<15}"
        separator = "-" * 70
        print(header)
        print(separator)
        f.write(header + "\n" + separator + "\n")

        for i, m in enumerate(inner_model.model):
            # We look for weights/biases in the standard locations
            # Note: Many YOLO layers wrap the actual Conv in a .conv attribute
            weight_shape = "None"
            bias_shape = "None"
            
            # Check for standard Conv layers
            target = None
            if hasattr(m, 'conv'):
                target = m.conv
            elif isinstance(m, torch.nn.Conv2d):
                target = m
                
            if target:
                if hasattr(target, 'weight') and target.weight is not None:
                    weight_shape = str(list(target.weight.shape))
                if hasattr(target, 'bias') and target.bias is not None:
                    bias_shape = str(list(target.bias.shape))
            
            # For complex blocks (C3k2), weights are inside sub-modules
            # We'll report if the block contains parameters
            elif sum(p.numel() for p in m.parameters()) > 0:
                weight_shape = "Complex Block"
                bias_shape = "See Sub-layers"

            line = f"{i:<5} | {m.__class__.__name__:<15} | {weight_shape:<20} | {bias_shape:<15}"
            print(line)
            f.write(line + "\n")

    print(f"\n✅ Weight shapes logged to: {log_filepath}")

# To get actual numerical values for a specific layer (e.g., Layer 0)
def get_raw_values(model, layer_idx):
    layer = model.model.model[layer_idx]
    if hasattr(layer, 'conv'):
        weights = layer.conv.weight.data
        bias = layer.conv.bias.data if layer.conv.bias is not None else None
        return weights, bias
    return None, None



import torch
import os
from ultralytics import YOLO

def extract_raw_parameters(model_path, log_dir="../logs"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filepath = os.path.join(log_dir, "raw_parameters_summary.log")
    model = YOLO(model_path)
    state_dict = model.model.state_dict()
    
    with open(log_filepath, "w") as f:
        f.write(f"RAW WEIGHT AND BIAS SUMMARY for {model_path}\n")
        f.write(f"{'Parameter Name':<50} | {'Shape':<20} | {'Mean':<10} | {'Std':<10}\n")
        f.write("-" * 100 + "\n")
        
        for name, param in state_dict.items():
            # Cast to float to avoid "Long" dtype errors during math
            p_float = param.detach().float() 
            
            mean_val = p_float.mean().item()
            std_val = p_float.std().item()
            shape_str = str(list(param.shape))
            
            line = f"{name:<50} | {shape_str:<20} | {mean_val:<10.6f} | {std_val:<10.6f}"
            f.write(line + "\n")
            
            # Print feedback to terminal for first few layers
            if any(x in name for x in ["model.0.", "model.1."]):
                print(line)

    print(f"\n✅ Detailed parameter stats saved to: {log_filepath}")



def export_parameters_to_csv(model_path, output_csv="../logs/yolo26_raw_params.csv"):
    # Ensure directory exists
    log_dir = os.path.dirname(output_csv)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load the model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    state_dict = model.model.state_dict()

    rows = []

    print("Extracting weights and biases... this may take a moment.")
    for name, param in state_dict.items():
        # Flatten the tensor to handle any dimension (1D, 2D, 4D)
        # Cast to float to ensure consistency
        flattened = param.detach().cpu().float().numpy().flatten()
        shape = str(list(param.shape))
        
        for i, value in enumerate(flattened):
            rows.append({
                "Layer_Name": name,
                "Shape": shape,
                "Element_Index": i,
                "Value": value
            })

    # Create DataFrame and Export
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Success! CSV saved to: {output_csv}")
    print(f"Total parameters exported: {len(df)}")



def export_each_param_to_csv(model_path, base_log_dir="../params"):
    # 1. Create a specific sub-directory for individual files
    target_dir = os.path.join(base_log_dir, "individual_params")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # 2. Load the model
    model = YOLO(model_path)
    state_dict = model.model.state_dict()

    print(f"Exporting {len(state_dict)} parameters to individual CSV files...")

    for name, param in state_dict.items():
        # Sanitize name for filename (e.g., model.0.conv.weight -> model_0_conv_weight.csv)
        clean_name = name.replace('.', '_')
        file_path = os.path.join(target_dir, f"{clean_name}.csv")
        
        # Extract raw data
        flattened_data = param.detach().cpu().float().numpy().flatten()
        
        # Create a simple DataFrame for this specific parameter
        df = pd.DataFrame({
            'Index': range(len(flattened_data)),
            'Value': flattened_data
        })
        
        # Save to individual CSV
        df.to_csv(file_path, index=False)
        
        # Optional: Print progress for key layers
        if "model_0" in clean_name:
            print(f"Saved: {clean_name}.csv | Elements: {len(flattened_data)}")

    print(f"\n✅ Done! All parameters are saved in: {target_dir}")


def export_raw_values_only(model_path, base_log_dir="../logs"):
    # 1. Create a specific sub-directory
    target_dir = os.path.join(base_log_dir, "individual_params_raw")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 2. Load the model
    model = YOLO(model_path)
    state_dict = model.model.state_dict()

    print(f"Exporting raw values to: {target_dir}")

    for name, param in state_dict.items():
        # Sanitize filename
        clean_name = name.replace('.', '_')
        file_path = os.path.join(target_dir, f"{clean_name}.csv")
        
        # Extract raw data and flatten
        # Ensure it's on CPU and converted to float32
        data = param.detach().cpu().float().numpy().flatten()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save without Header and without Index
        df.to_csv(file_path, index=False, header=False)

    print(f"✅ Export complete. Each CSV now contains only a single column of raw numbers.")
    




def quantize_and_extract():
    # 1. Load FP32 Model
    print("--- Step 1: Loading FP32 Model ---")
    model_wrapper = YOLO(FP_MODEL)
    model = model_wrapper.model.cpu()
    model.eval()

    # 2. Prepare for Static Quantization
    # We fuse layers to optimize internal INT8 math
    model.fuse()
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    # 3. Calibration with SAMPLE_IMAGE
    print(f"--- Step 2: Calibrating with {SAMPLE_IMAGE} ---")
    if os.path.exists(SAMPLE_IMAGE):
        img = cv2.imread(SAMPLE_IMAGE)
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1)) # HWC to CHW
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
    else:
        print("Warning: SAMPLE_IMAGE not found. Using random tensor for calibration.")
        img = torch.randn(1, 3, 640, 640)

    # Pass the image through the model to set activation scales
    with torch.no_grad():
        model(img)

    # 4. Convert and Save INT8 Model
    print("--- Step 3: Converting to INT8 ---")
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    if not os.path.exists(os.path.dirname(INT8_MODEL_PATH)):
        os.makedirs(os.path.dirname(INT8_MODEL_PATH))
    
    # Save the quantized state dict
    torch.save(quantized_model.state_dict(), INT8_MODEL_PATH)
    print(f"✅ Quantized model saved to: {INT8_MODEL_PATH}")

    # 5. Extract Raw INT8 Weights and Biases to CSV
    print("--- Step 4: Extracting Raw INT8 Values ---")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    for name, module in quantized_model.named_modules():
        # Check for quantized Conv or Linear layers
        if hasattr(module, 'weight') and callable(module.weight):
            q_weight = module.weight()
            
            # Check if it's a quantized tensor
            if q_weight.is_quantized:
                clean_name = name.replace('.', '_')
                
                # A. Export Weights (Raw Integers)
                weight_ints = q_weight.int_repr().numpy().flatten()
                pd.DataFrame(weight_ints).to_csv(
                    f"{LOG_DIR}/{clean_name}_weight.csv", index=False, header=False
                )
                
                # B. Export Bias (If exists - often stored as float32 in PTQ)
                if hasattr(module, 'bias') and module.bias() is not None:
                    bias_vals = module.bias().detach().numpy().flatten()
                    pd.DataFrame(bias_vals).to_csv(
                        f"{LOG_DIR}/{clean_name}_bias.csv", index=False, header=False
                    )

                # C. Export Quantization Parameters (Scale/Zero Point)
                with open(f"{LOG_DIR}/{clean_name}_meta.txt", "w") as f:
                    f.write(f"Scale: {q_weight.q_scale()}\n")
                    f.write(f"ZeroPoint: {q_weight.q_zero_point()}\n")

    print(f"✅ All INT8 parameters exported to {LOG_DIR}")


def export_integer_params(model_path, sample_img_path, log_dir="../logs/integer_params"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. Load and Quantize Model (Static PTQ)
    model_wrapper = YOLO(model_path)
    model = model_wrapper.model.cpu()
    model.eval()
    model.fuse()
    
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration to get Input Scales
    dummy_input = torch.randn(1, 3, 640, 640)
    model(dummy_input)
    
    q_model = torch.quantization.convert(model, inplace=False)

    print(f"Exporting integer weights and biases to {log_dir}...")

    for name, module in q_model.named_modules():
        # Target Quantized Convolution layers
        if hasattr(module, 'weight') and callable(module.weight):
            q_weight = module.weight()
            
            if q_weight.is_quantized:
                clean_name = name.replace('.', '_')
                
                # --- A. WEIGHTS (INT8) ---
                weight_int8 = q_weight.int_repr().numpy().flatten()
                pd.DataFrame(weight_int8).to_csv(
                    f"{log_dir}/{clean_name}_weight_int8.csv", index=False, header=False
                )

                # --- B. BIAS (INT32) ---
                if hasattr(module, 'bias') and module.bias() is not None:
                    # Get required scales
                    s_weight = q_weight.q_scale()
                    # PyTorch stores input scale in the activation observer
                    # If unavailable, we use a standard approximation or the module's scale
                    s_input = getattr(module, 'scale', 0.00392) # 1/255 approx
                    
                    bias_fp32 = module.bias().detach().cpu().numpy()
                    
                    # Quantize bias to INT32
                    bias_int32 = np.round(bias_fp32 / (s_input * s_weight)).astype(np.int32)
                    
                    pd.DataFrame(bias_int32.flatten()).to_csv(
                        f"{log_dir}/{clean_name}_bias_int32.csv", index=False, header=False
                    )

    print(f"✅ Finished. Weights are INT8, Biases are INT32.")


def quantize_and_export_all(fp32_path, sample_img_path, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 1. Load and Quantize (Keep the object in memory)
    print("--- Quantizing Model ---")
    model_wrapper = YOLO(fp32_path)
    model = model_wrapper.model.cpu()
    model.eval()
    model.fuse()
    
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration
    dummy_input = torch.randn(1, 3, 640, 640)
    model(dummy_input)
    
    # This is your active quantized model object
    q_model = torch.quantization.convert(model, inplace=False)

    # 2. Export directly from the object
    print(f"--- Exporting Integer Params to {log_dir} ---")
    
    for name, module in q_model.named_modules():
        if hasattr(module, 'weight') and callable(module.weight):
            q_weight = module.weight()
            
            if q_weight.is_quantized:
                clean_name = name.replace('.', '_')
                
                # --- WEIGHTS (INT8) ---
                weight_int8 = q_weight.int_repr().numpy().flatten()
                pd.DataFrame(weight_int8).to_csv(
                    f"{log_dir}/{clean_name}_weight_int8.csv", index=False, header=False
                )

                # --- BIAS (INT32) ---
                if hasattr(module, 'bias') and module.bias() is not None:
                    # Scales for quantization
                    s_weight = q_weight.q_scale()
                    s_input = 0.00392157  # Standard 1/255 for normalized images
                    
                    bias_fp32 = module.bias().detach().cpu().numpy()
                    
                    # Integer Bias Formula for Hardware
                    bias_int32 = np.round(bias_fp32 / (s_input * s_weight)).astype(np.int32)
                    
                    pd.DataFrame(bias_int32.flatten()).to_csv(
                        f"{log_dir}/{clean_name}_bias_int32.csv", index=False, header=False
                    )
    
    print("✅ Export successful.")


def check_int8_model_working(fp32_path, int8_weights_path, test_image_path):
    # 1. Reconstruct the quantized structure
    # We need the original architecture to 'hold' the quantized weights
    print("Preparing quantized architecture...")
    model_wrapper = YOLO(fp32_path)
    model = model_wrapper.model.cpu()
    model.eval()
    model.fuse()
    
    # Apply the same quantization config used during saving
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Convert to quantized version (this creates the 'quantized' layers)
    q_model = torch.quantization.convert(model, inplace=False)

    # 2. Load the INT8 state_dict
    print(f"Loading INT8 weights from {int8_weights_path}...")
    state_dict = torch.load(int8_weights_path)
    q_model.load_state_dict(state_dict)
    q_model.eval()

    # 3. Test Inference
    print(f"Running test inference on {test_image_path}...")
    if not os.path.exists(test_image_path):
        # Fallback to random data if image missing
        img_tensor = torch.randn(1, 3, 640, 640)
    else:
        img = cv2.imread(test_image_path)
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1)) / 255.0
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)

    with torch.no_grad():
        output = q_model(img_tensor)

    # 4. Verify Output
    # YOLO outputs are usually a list of tensors for different scales
    print("\n--- Inference Result ---")
    if isinstance(output, (list, tuple)):
        for i, out in enumerate(output):
            print(f"Scale {i} Output Shape: {out.shape}")
            print(f"Scale {i} Data Sample (first 5): {out.flatten()[:5]}")
    else:
        print(f"Output Shape: {output.shape}")

    print("\n✅ Verification complete. If the output contains non-zero numbers and correct shapes, your INT8 .pt file is working correctly.")



def extract_weights_from_onnx(onnx_path, log_dir="../logs/onnx_int8_extraction"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Load the ONNX model
    model = onnx.load(onnx_path)
    weights = model.graph.initializer

    print(f"Found {len(weights)} weight/bias tensors in ONNX graph.")

    for tensor in weights:
        # Convert ONNX tensor to NumPy array
        name = tensor.name.replace("/", "_").replace(".", "_")
        raw_data = numpy_helper.to_array(tensor)
        
        # Flatten and save
        flattened = raw_data.flatten()
        
        # Check if data is actually INT8/UINT8 or FLOAT
        dtype_str = str(raw_data.dtype)
        
        file_path = os.path.join(log_dir, f"{name}_{dtype_str}.csv")
        pd.DataFrame(flattened).to_csv(file_path, index=False, header=False)
        
        # Log the shape and type for your report
        print(f"Extracted: {name} | Shape: {raw_data.shape} | Type: {dtype_str}")

    print(f"✅ All ONNX initializers exported to {log_dir}")


def extract_from_openvino(xml_path, bin_path, log_dir="../logs/openvino_extraction"):
    # 1. Initialize OpenVINO Core
    core = ov.Core()
    
    # 2. Read the network
    # This parses the XML and maps it to the binary weights in the .bin file
    model = core.read_model(model=xml_path, weights=bin_path)
    
    os.makedirs(log_dir, exist_ok=True)
    print(f"--- Extracting from OpenVINO IR: {xml_path} ---")

    # 3. Iterate through operations
    for op in model.get_ops():
        # OpenVINO IR stores all weights and biases as 'Constant' nodes
        if op.get_type_name() == "Constant":
            # Sanitize the name for Linux filenames
            clean_name = op.get_friendly_name().replace("/", "_").replace(".", "_")
            
            # Get the actual numeric data
            data = op.get_data()
            
            # Identify the type for your MTP records
            # Usually: 4D = Weights (OIHW), 1D = Bias or Scale
            param_type = "weight" if len(data.shape) > 1 else "bias_or_scale"
            dtype_str = str(data.dtype)

            # 4. Save to CSV
            file_name = f"{clean_name}_{param_type}_{dtype_str}.csv"
            save_path = os.path.join(log_dir, file_name)
            
            pd.DataFrame(data.flatten()).to_csv(save_path, index=False, header=False)
            
            print(f"Extracted: {clean_name} | Shape: {data.shape} | Dtype: {dtype_str}")

    print(f"\n✅ Done! All parameters are in: {log_dir}")

# Example usage:
# Yolo_Detect_FP(FP_MODEL)
# get_yolo26_details(FP_MODEL)
# extract_layer_weights(FP_MODEL)
# extract_raw_parameters(FP_MODEL)
# export_parameters_to_csv(FP_MODEL)
# export_raw_values_only(FP_MODEL)
# quantize_and_extract()
# export_integer_params(INT8_MODEL_PATH,SAMPLE_IMAGE , LOG_DIR)
# quantize_and_export_all(FP_MODEL, SAMPLE_IMAGE, LOG_DIR)
# check_int8_model_working(FP_MODEL, INT8_MODEL_PATH, SAMPLE_IMAGE)

# export_and_quantize_onnx("../models/yolo26n.pt")


extract_from_openvino("../models/yolo26n.xml", "../models/yolo26n.bin")
