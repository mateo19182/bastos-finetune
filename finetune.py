import json
import logging
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import time
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def log_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

# Custom callback for logging
class DebugCallback:
    def __init__(self):
        self.step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.step += 1
        logger.info(f"Step {self.step}: loss = {logs.get('loss', 'N/A')}")
        log_memory_usage()

# Main execution
if __name__ == "__main__":
    logger.info("Starting script execution")
    
    # Load the dataset
    file_path = 'bastos/bastos-llama.txt'
    dataset = load_preformatted_data(file_path)
    logger.info(f"Dataset loaded. Size: {len(dataset)}")

    # Model initialization
    max_seq_length = 2048
    dtype = torch.float32  # Use full precision instead of quantization

    logger.info("Initializing model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
    )
    logger.info("Model initialized")

    logger.info("Setting up PEFT model...")
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
    logger.info("PEFT model setup complete")

    # Set up the trainer
    logger.info("Setting up trainer...")
    debug_callback = DebugCallback()
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
            max_steps = 80,
            learning_rate = 5e-4,
            fp16 = False,
            bf16 = False,
            logging_steps = 1,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
        callbacks=[debug_callback],
    )
    logger.info("Trainer setup complete")

    # Train the model
    logger.info("Starting training...")
    start_time = time.time()
    try:
        trainer_stats = trainer.train()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
    finally:
        end_time = time.time()
        logger.info(f"Training time: {end_time - start_time:.2f} seconds")

    # Save the model
    logger.info("Saving the model...")
    try:
        model.save_pretrained("lora_model")
        tokenizer.save_pretrained("lora_model")
        model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"An error occurred while saving the model: {str(e)}")

    # Test the model
    logger.info("Testing the model...")
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "user", "content": "Ideas de Miguel Anxo Bastos sobre el sistema educativo"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt",
    )
    input_ids = inputs.to("cuda")

    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    logger.info("Generating text...")
    try:
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            streamer = text_streamer, 
            max_new_tokens = 2056,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            temperature = 0.7,
            top_p = 0.95
        )
        logger.info("Text generation complete")
    except Exception as e:
        logger.error(f"An error occurred during text generation: {str(e)}")

    # Print the full generated text without streaming
    logger.info("Full generated text:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

logger.info("Script execution completed.")