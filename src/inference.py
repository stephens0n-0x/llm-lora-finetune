import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import load_config


def load_model_for_inference(config: dict):
    """
    Load the base model and merge the LoRA adapter on top.

    Important distinction from training:
      - During training: base model frozen, adapter trained separately
      - During inference: we MERGE the adapter into the base model weights
        so there is zero extra latency. The merged model behaves identically
        to a fully fine-tuned model but was much cheaper to produce.
    """
    model_name = config["model"]["name"]
    adapter_path = config["training"]["output_dir"]
    dtype = getattr(torch, config["model"]["torch_dtype"])

    print(f"[inference] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"[inference] Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("[inference] Adapter merged into base model")
    else:
        print("[inference] No adapter found — running base model only")

    model.eval()
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    user_message: str,
    config: dict,
    max_new_tokens: int = 256,
) -> str:
    """
    Generate a single response to a user message.

    We format the input using the same ChatML template used during training.
    Consistency between training format and inference format is critical —
    a mismatch here is one of the most common causes of poor inference quality.
    """
    # Format input using the same ChatML template from data_utils.py
    prompt = (
        f"<|im_start|>user\n{user_message}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    response = response.replace("<|im_end|>", "").strip()

    return response


def chat(config_path: str):
    """
    Interactive chat loop. Keeps a conversation history so the model
    has context from previous messages — this is what makes it feel
    like a real assistant rather than a single-turn Q&A system.
    """
    config = load_config(config_path)
    model, tokenizer = load_model_for_inference(config)

    print("\n" + "=" * 50)
    print("  Chat with your fine-tuned model")
    print("  Type 'quit' to exit, 'reset' to clear history")
    print("=" * 50 + "\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Exiting.")
            break
        if user_input.lower() == "reset":
            conversation_history = []
            print("[history cleared]\n")
            continue

        if conversation_history:
            recent = conversation_history[-3:]
            history_text = ""
            for turn in recent:
                history_text += (
                    f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
                    f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
                )
            # Append current user message after history
            full_input = history_text + user_input
        else:
            full_input = user_input

        response = generate_response(model, tokenizer, full_input, config)

        conversation_history.append({
            "user": user_input,
            "assistant": response,
        })

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lora_config.yaml")
    chat(config_path)