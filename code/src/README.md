To execute the code, first run training.py to train the model and then run_prompt.py to execute the pipeline.

python training.py - Execute this first
python run_prompt.py - Then execute this.

This code is a complete pipeline for fine-tuning a pre-trained BERT model for a text classification task. 
It demonstrates the process of preparing a dataset, tokenizing text, converting it into a PyTorch-compatible format, training the model,
and saving the fine-tuned model for future use.