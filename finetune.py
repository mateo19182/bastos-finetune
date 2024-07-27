import json
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb


# Define the chat template
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM} <|eot_id|><|start_header_id|>user<|end_header_id|>

{INPUT} <|eot_id|><|start_header_id|>assistant<|end_header_id|>

{OUTPUT} <|eot_id|>"""

# Function to load and process JSONL data
def load_jsonl_for_training(file_path, chat_template):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            messages = item['messages']
            
            system = next((msg['content'] for msg in messages if msg['role'] == 'system'), "")
            user = next((msg['content'] for msg in messages if msg['role'] == 'user'), "")
            assistant = next((msg['content'] for msg in messages if msg['role'] == 'assistant'), "")
            
            formatted_text = chat_template.format(
                SYSTEM=system,
                INPUT=user,
                OUTPUT=assistant
            )
            data.append({"text": formatted_text})
    
    return Dataset.from_list(data)

def load_preformatted_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: 
        data = f.read().split('\n\n')  # Split by double newline to separate examples
    return Dataset.from_dict({"text": data})

# Main execution
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="bastos", config={
        "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.01,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "use_rslora": False,
    })

    # Load the dataset
    file_path = 'bastos/bastos-llama.txt'
    dataset = load_preformatted_data(file_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Log dataset info
    wandb.log({"dataset_size": len(dataset)})

    # Model initialization
    max_seq_length = 4096
    dtype = None
    load_in_4bit = False

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # Log model architecture
    wandb.watch(model, log="all", log_freq=10)

    # Set up the trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 200,
            learning_rate = 5e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb",  # Enable wandb logging in TrainingArguments
        ),
    )

    # Train the model
    print("Starting training...")
    trainer_stats = trainer.train()
    print("Training completed.")

    # Log final training stats
    wandb.log({
        #"final_loss": trainer_stats.loss,
        "total_steps": trainer_stats.global_step,
        #"total_time": trainer_stats.total_flos,
    })

    # Save the model
    print("Saving the model...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
    print("Model saved.")

    # Test the model
    print("Testing the model...")
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": "Eres Miguel Anxo Bastos dando una conferencia. "},
        {"role": "user", "content": "ideas de Miguel Anxo Bastos sobre la escuela austriaca"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    )
    input_ids = inputs.to("cuda")

    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    print("Generated text:")
    output = model.generate(
        input_ids, 
        attention_mask=attention_mask,
        streamer = text_streamer, 
        max_new_tokens = 2048,
        pad_token_id = tokenizer.eos_token_id,
        do_sample = True,
        temperature = 0.7,
        top_p = 0.95
    )

    # Log generated text
    wandb.log({"generated_text": tokenizer.decode(output[0], skip_special_tokens=True)})

    # Print the full generated text without streaming
    print("\nFull generated text:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # Finish the wandb run
    wandb.finish()

print("Script execution completed.")