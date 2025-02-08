from datasets import load_dataset
from utils import BoxQuantizer
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import os 
from torch.utils.data import DataLoader
from tqdm import tqdm 
from transformers import AdamW, get_scheduler

data = load_dataset("agent-studio/GroundUI-18K")

num_bbox_height_bins = 1000
num_bbox_width_bins = 1000
box_quantization_mode = 'floor'
box_quantizer = BoxQuantizer(
    box_quantization_mode,
    (num_bbox_width_bins, num_bbox_height_bins),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base-ft",
    trust_remote_code=True,
).to(device)

processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base-ft", 
    trust_remote_code=True
)

for param in model.vision_tower.parameters():
  param.is_trainable = False

class Dataset(Dataset): 
    def __init__(self, data): 
        self.data = data
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, idx):
        example = self.data[idx]
        image = example['image'].convert("RGB")
        question = "<OPEN_VOCABULARY_DETECTION>" + example['instruction']
        quantized_bbox = box_quantizer.quantize(boxes=torch.tensor(example['bbox']), size=example['resolution'])
        quantized_string = ''.join([f'<loc_{int(x)}>' for x in quantized_bbox])
        answer = example['instruction'] + quantized_string
        return question, answer, image

def collate_fn(batch): 
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(device)
    return inputs, answers

train_dataset = Dataset(data['train'])
batch_size = 12
num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                          collate_fn=collate_fn, num_workers=num_workers, shuffle=True)

epochs = 7
optimizer = AdamW(model.parameters(), lr=1e-6)
num_training_steps = epochs * len(train_loader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, 
                              num_warmup_steps=0, num_training_steps=num_training_steps)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    running_loss = 0 
    i = -1
    
    for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
        i += 1
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        labels = processor.tokenizer(text=answers, return_tensors="pt", padding=True, return_token_type_ids=False).input_ids.to(device)
        
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        current_loss = loss.item()
        train_loss += current_loss
        running_loss += current_loss
        
        if (i + 1) % 250 == 0:
            avg_running_loss = running_loss / 1000
            print(f"Epoch {epoch + 1}, Step {i + 1}, Average Loss over last 1000 steps: {avg_running_loss:.4f}")
            running_loss = 0
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1} Complete - Average Training Loss: {avg_train_loss:.4f}")

    output_dir = f"./model_checkpoints_groundUi18k/epoch_{epoch+1}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)