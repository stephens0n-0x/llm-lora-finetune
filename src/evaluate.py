import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data_utils import load_config

# A small fixed set of test questions to compare base vs fine-tuned
TEST_PROMPTS = [
    "What is the difference between supervised and unsupervised learning?",
    "Write a short poem about the ocean.",
    "Explain what a neural network is to a 10 year old.",
    "What are three benefits of exercise?",
    "Summarize what machine learning is in two sentences.",
]


def load_base_model(config: dict):
    """Load the original pretrained model with no adapter."""
    model_name = config["model"]["name"]
    dtype = getattr(torch, config["model"]["torch_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(config: dict):
    """Load the base model with the LoRA adapter merged in."""
    model_name = config["model"]["name"]
    adapter_path = config["training"]["output_dir"]
    dtype = getattr(torch, config["model"]["torch_dtype"])
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response using the model's native chat template."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(formatted, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for evaluation — deterministic and reproducible
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_config.yaml")
    config = load_config(config_path)

    print("Loading base model...")
    base_model, base_tokenizer = load_base_model(config)

    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_finetuned_model(config)

    print("\n" + "=" * 70)
    print("BASE vs FINE-TUNED — Side by Side Evaluation")
    print("=" * 70)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] Prompt: {prompt}")
        print("-" * 70)

        base_response = generate(base_model, base_tokenizer, prompt)
        print(f"BASE MODEL:\n{base_response}")
        print()

        ft_response = generate(ft_model, ft_tokenizer, prompt)
        print(f"FINE-TUNED:\n{ft_response}")
        print("=" * 70)

    # Free memory between models
    del base_model
    del ft_model
    torch.mps.empty_cache() if torch.backends.mps.is_available() else None


if __name__ == "__main__":
    evaluate()