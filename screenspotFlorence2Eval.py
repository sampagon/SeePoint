#!/usr/bin/env python
import os
import random
import argparse
from datasets import load_dataset
from PIL import Image, ImageDraw
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

def normalize_coords(bbox, image_width, image_height):
    """Convert absolute coordinates to normalized coordinates (0-1)."""
    return [
        bbox[0] / image_width,
        bbox[1] / image_height,
        bbox[2] / image_width,
        bbox[3] / image_height
    ]

def run_example(model, processor, device, task_prompt, text_input, image):
    """
    Run inference using the Florence model.

    Args:
        model: The loaded model.
        processor: The associated processor.
        device: The torch device.
        task_prompt (str): The task prompt (e.g., "<OPEN_VOCABULARY_DETECTION>").
        text_input (str): The instruction text.
        image (PIL.Image): The image to process.

    Returns:
        parsed_answer (dict): The parsed answer (expected to contain a key 
                              "<OPEN_VOCABULARY_DETECTION>" with 'bboxes').
    """
    prompt = task_prompt + text_input

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer

def evaluate_model_binary(model, processor, dataset, device):
    """
    Evaluate the Florence model on the dataset on a binary pass/fail basis.

    For each sample, the predicted bounding box is extracted, its center is computed,
    and the prediction is considered a pass if the center falls within the ground-truth bounding box.

    Args:
        model: The loaded Florence model.
        processor: The associated processor.
        dataset: The dataset (expects each sample to have 'bbox', 'instruction', 'image').
        device: The device to run the model on.

    Returns:
        results (dict): Contains the total number of samples and number of passes.
    """
    results = {
        'total_samples': 0,
        'passes': 0
    }
    
    for sample in tqdm(dataset, desc="Evaluating"):
        results['total_samples'] += 1
        gt_bbox_norm = sample['bbox']  # Ground truth bbox in normalized coordinates: [x1, y1, x2, y2]
        gt_instruction = sample['instruction']
        image = sample['image']
        
        try:
            detection_results = run_example(model, processor, device, "<OPEN_VOCABULARY_DETECTION>", gt_instruction, image)
            # Extract predicted bounding box (in absolute coordinates)
            pred_bbox = detection_results['<OPEN_VOCABULARY_DETECTION>']['bboxes'][0]
            # Normalize the predicted bbox
            pred_bbox_norm = normalize_coords(pred_bbox, image.width, image.height)
            # Compute the center in normalized coordinates
            center_x = (pred_bbox_norm[0] + pred_bbox_norm[2]) / 2
            center_y = (pred_bbox_norm[1] + pred_bbox_norm[3]) / 2
            
            # Check whether the predicted center falls within the ground truth bbox.
            if gt_bbox_norm[0] <= center_x <= gt_bbox_norm[2] and gt_bbox_norm[1] <= center_y <= gt_bbox_norm[3]:
                results['passes'] += 1
        except Exception as e:
            # Optionally log the error here.
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Florence Model Evaluation Script")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cpu' or 'cuda'). If not set, uses cuda if available."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["base", "large"],
        default="base",
        help="Choose model size: 'base' or 'large'. This sets the pretrained identifier for config and processor."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save visualization images."
    )
    parser.add_argument(
        "--num_visualization",
        type=int,
        default=10,
        help="Number of random examples to visualize."
    )
    args = parser.parse_args()

    # Set up device.
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Select the pretrained identifier based on the model size.
    if args.model_size.lower() == "base":
        pretrained_model = "microsoft/Florence-2-base-ft"
    elif args.model_size.lower() == "large":
        pretrained_model = "microsoft/Florence-2-large-ft"
    else:
        raise ValueError("Invalid model size specified. Choose 'base' or 'large'.")

    # Load model configuration, checkpoint, and processor.
    print("Loading model configuration...")
    base_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
    print("Loading model checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=base_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(pretrained_model, trust_remote_code=True)

    # Load the dataset.
    print("Loading dataset...")
    ds = load_dataset("rootsautomation/ScreenSpot")
    test_dataset = ds['test']

    # Run the binary evaluation.
    print("Running binary evaluation...")
    binary_results = evaluate_model_binary(model, processor, test_dataset, device)
    total = binary_results['total_samples']
    passes = binary_results['passes']
    pass_rate = passes / total if total > 0 else 0

    print("\nBinary Evaluation Results (Based on Predicted Center):")
    print(f"Total samples: {total}")
    print(f"Passes: {passes}")
    print(f"Pass rate: {pass_rate:.4f}")

    # Generate visualizations for a few random examples.
    print("Generating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    indices = random.sample(range(len(test_dataset)), min(args.num_visualization, len(test_dataset)))

    for idx in indices:
        sample = test_dataset[idx]
        gt_bbox_norm = sample['bbox']  # Ground truth bbox (normalized)
        instruction = sample['instruction']
        image = sample['image']
        
        try:
            detection_results = run_example(model, processor, device, "<OPEN_VOCABULARY_DETECTION>", instruction, image)
            pred_bbox = detection_results['<OPEN_VOCABULARY_DETECTION>']['bboxes'][0]
            pred_bbox_norm = normalize_coords(pred_bbox, image.width, image.height)
            center_x = (pred_bbox_norm[0] + pred_bbox_norm[2]) / 2
            center_y = (pred_bbox_norm[1] + pred_bbox_norm[3]) / 2
            passed = (gt_bbox_norm[0] <= center_x <= gt_bbox_norm[2] and gt_bbox_norm[1] <= center_y <= gt_bbox_norm[3])
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

        # Convert ground truth bbox from normalized to absolute coordinates.
        gt_bbox_abs = [
            int(gt_bbox_norm[0] * image.width),
            int(gt_bbox_norm[1] * image.height),
            int(gt_bbox_norm[2] * image.width),
            int(gt_bbox_norm[3] * image.height)
        ]
        draw = ImageDraw.Draw(image)
        # Draw the ground truth bbox in yellow.
        draw.rectangle(gt_bbox_abs, outline="yellow", width=3)
        # Draw the predicted center as a red circle.
        r = 10  # circle radius
        abs_center = [int(center_x * image.width), int(center_y * image.height)]
        draw.ellipse(
            [abs_center[0]-r, abs_center[1]-r, abs_center[0]+r, abs_center[1]+r],
            fill="red"
        )
        
        label = "PASS" if passed else "FAIL"
        output_path = os.path.join(args.output_dir, f"sample_{idx}_{label}.png")
        image.save(output_path)
        print(f"Saved sample {idx} as {label} to {output_path}")

if __name__ == "__main__":
    main()
