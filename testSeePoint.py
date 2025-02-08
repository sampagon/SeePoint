#!/usr/bin/env python
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image, ImageDraw

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_boxes_on_image(image, boxes, labels, output_path):
    """
    Draw bounding boxes and labels on the image and save it.
    """
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='black', width=3)
    
    image_with_boxes.save(output_path)
    return image_with_boxes

def run_example(model, processor, task_prompt, text_input, image):
    """
    Run inference with the Florence2 model on an image with the given query.
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

def main():
    parser = argparse.ArgumentParser(
        description="Run Florence2 UI element detection on an image."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path or Hugging Face repo ID of the model checkpoint."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text for detection (e.g., 'Click the paste button')."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output image with bounding boxes."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["base", "large"],
        default="base",
        help="Model size to use ('base' or 'large'). Use 'large' for Florence2-large checkpoints."
    )
    args = parser.parse_args()

    # Select the correct configuration based on the model size.
    if args.model_size == "large":
        config_name = "microsoft/Florence-2-large-ft"
    else:
        config_name = "microsoft/Florence-2-base-ft"
    
    # Load the configuration
    base_config = AutoConfig.from_pretrained(config_name, trust_remote_code=True)

    # Load the model checkpoint using the appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=base_config,
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    # Load the processor
    processor = AutoProcessor.from_pretrained(config_name, trust_remote_code=True)

    # Open the input image
    image = Image.open(args.image)

    # Run inference
    detection_results = run_example(model, processor, "<OPEN_VOCABULARY_DETECTION>", args.query, image)

    # Process and draw results if available
    if '<OPEN_VOCABULARY_DETECTION>' in detection_results:
        results = detection_results['<OPEN_VOCABULARY_DETECTION>']
        if results['bboxes']:
            draw_boxes_on_image(image, results['bboxes'], results['bboxes_labels'], args.output_path)
            print(f"Image saved with bounding boxes at: {args.output_path}")
        else:
            print("No bounding boxes detected!")
    else:
        print("Detection results do not contain '<OPEN_VOCABULARY_DETECTION>' key.")

if __name__ == "__main__":
    main()

