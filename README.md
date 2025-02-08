# SeePoint

SeePoint is a lightweight, open vocabulary, pixel coordinate predictor for UI elements. Given a natural language query—such as "click on the firefox icon"—and a screenshot of a user interface, SeePoint returns the pixel coordinates corresponding to the target UI element.

## Overview

SeePoint currently leverages finetuned variants of Microsoft's Florence2 model to accurately predict the location of UI elements based on open-vocabulary queries. The system has been finetuned on two datasets:

- **Wave-UI Dataset:**  
  - **Descriptions Only:** 
  - **Instruction, Name, and Description (Even Split):**

- **GroundUI-18k Dataset:**  
  - Finetuned using both the Florence2 Base and Florence2 Large models, with Florence2 Large (trained for 3 epochs) providing significantly better results than Florence2 Base.

## Models & Training

We have trained several model variants:

- **WaveUI-Description-Florence2Base (6 epochs)**
- **WaveUI-InstructionNameDescription-Florence2Base (7 epochs)**
- **GroundUI18k-Florence2Base (7 epochs)**
- **GroundUI18k-Florence2Large (3 epochs)**

The current evaluation results on the Screenspot metric are:

| Model                                                       | Screenspot Score | Predictions/sec |
|-------------------------------------------------------------|------------------|-----------------|
| WaveUI-Description-Florence2Base (6 epochs)                 | 0.34             | 5               |
| WaveUI-InstructionNameDescription-Florence2Base (7 epochs)  | 0.46             | 5               |
| GroundUI18k-Florence2Base (7 epochs)                          | 0.40             | 5               |
| GroundUI18k-Florence2Large (3 epochs)                         | 0.56             | 3               |

**Note:** Based on these results, finetuning Florence2 Large on the WaveUI dataset using the instruction/name/description split is expected to yield Screenspot scores between 0.6 and 0.7—though this experiment has not been run yet.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/SeePoint.git
   cd SeePoint

2. **Install the dependencies**
   
   Everything was tested using Python 3.11
   ```bash
   pip install -r requirements.txt
   ```

## Running Evaluation
   
   ```bash
   python screenspotFlorence2Eval.py --device cuda --model_size (base or large) --checkpoint /path/to/your/checkpoint --output_dir /path/to/your/output_dir --num_visualization 10
   ```

## Future Work

   Just trying to make this thing as lightweight, fast, and accurate as possible...
   This will probably require better datasets and better architectures, but you gotta start somewhere.