---
license: mit
task_categories:
- text-to-audio
language:
- en
---
# TTA-Bench Dataset

##  🎯 Overview 
Welcome to TTA-Bench! This repository contains our comprehensive evaluation framework for text-to-audio (TTA) systems. We've carefully curated 2,999 prompts across six different evaluation dimensions, creating a standardized benchmark for assessing text-to-audio generation capabilities.

## 📚 Dataset Structure

Each prompt in our dataset contains these essential fields:
- `id`: Unique identifier for each prompt (format: prompt_XXXX)
- `prompt_text`: The actual content of the prompt
- `Dimensions of evaluation`: The evaluation dimension the prompt belongs to
- `source`: Origin of the prompt
- `notes`: Additional information and supplementary notes

## 🌟 Evaluation Dimensions

### 1. Accuracy (prompt_0001 - prompt_1500) 
Evaluates the precise representation of sound events and their relationships.
```json
{
    "id": "prompt_0001",
    "prompt_text": "...",
    "event_count": <number of events>,
    "event_list": ["event1", "event2", ...],
    "event_relation": "<relationship type>"
}
```

### 2. Generalization (prompt_1501 - prompt_1800) 
Tests the system's ability to handle novel and creative sound descriptions.

### 3. Robustness (prompt_1801 - prompt_2100) 
Assesses system performance under various text perturbations.

### 4. Fairness (prompt_2101 - prompt_2400) 
Evaluates bias and fairness aspects in audio generation. The prompts in this dimension are tagged with demographic attributes to assess potential biases:
```json
{
    "id": "prompt_XXXX",
    "prompt_text": "...",
    "Dimensions of evaluation": "Fairness",
    "notes": "<demographic_tag>"  // Contains one of: gender <male, female>, age<old, middle, youth, child>, or language <en, zh, other> tags
}
```

**Demographic Categories:**
- Gender: Evaluates gender-related biases in audio generation
- Age: Assesses age-related biases in generated content
- Language: Tests fairness across different language backgrounds

PS: Number 2325 is empty

### 5. Bias (prompt_2401 - prompt_2700) 
Examines potential biases in audio generation systems.

### 6. Toxicity (prompt_2701 - prompt_3000)
Assesses system responses to potentially harmful or inappropriate content.

Toxicity prompts include:
```json
{
    "id": "prompt_XXXX",
    "prompt_text": "...",
    "categories": ["category1", "category2", ...],
    "notes": {
        "source_prompt": "original prompt source"
    }
}
```
The categories inclue five distinct types:
- Hate
- Violence & Self-harm
- Sexual
- Shocking
- Illegal Activity

# 📋 Usage Guidelines
This toxicity part of dataset is intended solely for research use in evaluating the robustness and safety of text-to-text models against potentially toxic behavior. While the input prompts in this dataset are not explicitly harmful, they may induce undesirable outputs in some models. 

**Permitted Uses:** ✅
- Safety evaluation and robustness testing of TTA models
- Academic or non-commercial research related to content moderation, alignment, or adversarial prompting

**Prohibited Uses:** ❌
- Use of this data to train or fine-tune generative models without proper safety filtering
- Any commercial or production deployment involving toxic or harmful content
- Any use intended to produce, propagate, or reinforce hate speech, abuse, or offensive content

**Disclaimer:** 
The authors are not responsible for any misuse of the data. Users are expected to comply with applicable laws and ethical standards.

## ⚠️ Warning
Some prompts in the toxicity section may contain disturbing or inappropriate content. These are included solely for system evaluation purposes and should be handled with appropriate caution and professional context.

## 📜 License
MIT License

Copyright (c) 2024 TTA-Bench Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this dataset and associated documentation files (the "Dataset"), to deal
in the Dataset without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Dataset, and to permit persons to whom the Dataset is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Dataset.

The Dataset includes or is derived from the following third-party resources:

1. **AudioCaps Dataset**
   - Copyright (c) 2019 AudioCaps Team
   - URL: https://github.com/cdjkim/audiocaps
   - License: MIT License
   - Usage: This dataset includes portions of AudioCaps data, some of which are used directly and others which have been adapted or rewritten for the purposes of benchmark construction. All such uses comply with the original license terms, and the copyright of the AudioCaps Team is acknowledged and retained.

2. **I2P Dataset**
   - Copyright (c) 2023 AIML-TUDA Team
   - URL: https://huggingface.co/datasets/AIML-TUDA/i2p
   - License: MIT License
   - Usage: Portions of the I2P dataset were adapted and rewritten to better align with the design goals of our benchmark. The rewritten content retains the original MIT License, and the original authors are properly credited.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE
DATASET.