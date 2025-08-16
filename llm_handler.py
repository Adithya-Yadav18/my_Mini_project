# llm_handler.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use caching to load the model and tokenizer only once
@st.cache_resource
def load_model():
    """
    Loads the pre-trained IBM Granite model and tokenizer from Hugging Face.
    """
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Requires the 'accelerate' library
    )
    return tokenizer, model

def create_prompt(text, tone):
    """
    Creates a very specific and constrained instruction-based prompt for the model.
    """
    # *FIX:* Made the instructions much stricter to preserve the original meaning and content.
    base_instruction = "Your only task is to rewrite the 'Original Text' below in the specified tone. You MUST NOT add any new information, facts, or events. Preserve the core meaning exactly. The length of your output should be very similar to the original."

    prompts = {
        "Neutral": f"Rewrite the following text in a clear, objective, and neutral tone. {base_instruction}\n\nOriginal Text: \"{text}\"\n\nRewritten Text:",
        "Suspenseful": f"Transform the following text into a suspenseful and thrilling narrative. Build tension and a sense of mystery using only the details from the original text. {base_instruction}\n\nOriginal Text: \"{text}\"\n\nRewritten Text:",
        "Inspiring": f"Rewrite the following text to be inspiring and motivational. Use powerful and uplifting language to enhance the original message. {base_instruction}\n\nOriginal Text: \"{text}\"\n\nRewritten Text:"
    }
    return prompts.get(tone, prompts["Neutral"])

def rewrite_text(text, tone):
    """
    Rewrites the input text using the IBM Granite model with tightly controlled generation.
    """
    try:
        tokenizer, model = load_model()
        
        prompt = create_prompt(text, tone)
        
        model_inputs = tokenizer(prompt, return_tensors="pt")

        # *FIX:* Tightly control the output length to prevent new content.
        # Allows for a small expansion for stylistic words, but not whole new sentences.
        input_token_length = model_inputs["input_ids"].shape[1]
        max_new = int(input_token_length * 1.2) + 20 # Allow for 20% expansion + 20 tokens for safety

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=0.8, # Reduced top_p for more focused output
            temperature=0.6, # Lowered temperature to reduce randomness
            repetition_penalty=1.15, # Slightly increased to avoid repetition
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens for a cleaner output.
        input_length = model_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0, input_length:]
        rewritten_part = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return rewritten_part

    except Exception as e:
        st.error(f"An error occurred during text rewriting: {e}")
        print(f"Error details: {e}")
        return "Failed to rewrite text. Please check the logs."
