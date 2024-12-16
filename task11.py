from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import gc
import bitsandbytes as bnb

# Load the model and tokenizer
model_name = "gpt2"  
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def naive_inference(input_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=100)  # Max tokens can be adjusted
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, inference_time

def kv_cache_inference(input_text, model, tokenizer, device):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=100, use_cache=True)
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text, inference_time

def quantize_model_half(model, dtype=torch.float16):
    
    model = model.half() if dtype == torch.float16 else model
    return model

def quantize_model_with_bitsandbytes(model, quantization="8bit"):
    if quantization == "8bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
    elif quantization == "4bit":
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")
    else:
        raise ValueError("Unsupported quantization type. Use '8bit' or '4bit'.")
    return model

def measure_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved
    return 0, 0

def measure_inference_throughput(input_text, model, tokenizer, device, use_kv_cache=False, quantization=None, dtype=torch.float16):
    clear_cache()

    # Apply quantization if specified
    if quantization == "half":
        model = quantize_model_half(model, dtype=dtype)
    elif quantization:
        model = quantize_model_with_bitsandbytes(model, quantization=quantization)
    
    if use_kv_cache:
        generated_text, inference_time = kv_cache_inference(input_text, model, tokenizer, device)
    else:
        generated_text, inference_time = naive_inference(input_text, model, tokenizer, device)
    
    num_tokens = len(tokenizer.encode(generated_text))  # Count generated tokens
    throughput = num_tokens / inference_time  # Tokens per second

    allocated_memory, reserved_memory = measure_memory()
    
    return throughput, allocated_memory, reserved_memory, generated_text, inference_time

input_text = "Discuss the ethical implications of gene editing technologies like CRISPR, focusing on its potential impact on biodiversity."

# Baseline performance
throughput_baseline, memory_alloc_baseline, memory_reserved_baseline, generated_text_baseline, time_baseline = measure_inference_throughput(
    input_text, model, tokenizer, device, use_kv_cache=False, quantization=None)

# KV-cache performance
throughput_kv_cache, memory_alloc_kv_cache, memory_reserved_kv_cache, generated_text_kv_cache, time_kv_cache = measure_inference_throughput(
    input_text, model, tokenizer, device, use_kv_cache=True, quantization=None)

# 8-bit Quantized performance
throughput_8bit, memory_alloc_8bit, memory_reserved_8bit, generated_text_8bit, time_8bit = measure_inference_throughput(
    input_text, model, tokenizer, device, use_kv_cache=False, quantization="8bit")

# 4-bit Quantized performance
throughput_4bit, memory_alloc_4bit, memory_reserved_4bit, generated_text_4bit, time_4bit = measure_inference_throughput(
    input_text, model, tokenizer, device, use_kv_cache=False, quantization="4bit")

# Half precision (float16) Quantized performance
throughput_half, memory_alloc_half, memory_reserved_half, generated_text_half, time_half = measure_inference_throughput(
    input_text, model, tokenizer, device, use_kv_cache=False, quantization="half", dtype=torch.float16)

# Print the results
print(f"Baseline throughput: {throughput_baseline:.2f} tokens/sec, Allocated Memory: {memory_alloc_baseline} bytes, Reserved Memory: {memory_reserved_baseline} bytes, Time: {time_baseline:.2f}s")
print(f"KV-cache throughput: {throughput_kv_cache:.2f} tokens/sec, Allocated Memory: {memory_alloc_kv_cache} bytes, Reserved Memory: {memory_reserved_kv_cache} bytes, Time: {time_kv_cache:.2f}s")
print(f"8-bit Quantized throughput: {throughput_8bit:.2f} tokens/sec, Allocated Memory: {memory_alloc_8bit} bytes, Reserved Memory: {memory_reserved_8bit} bytes, Time: {time_8bit:.2f}s")
print(f"4-bit Quantized throughput: {throughput_4bit:.2f} tokens/sec, Allocated Memory: {memory_alloc_4bit} bytes, Reserved Memory: {memory_reserved_4bit} bytes, Time: {time_4bit:.2f}s")
print(f"Half precision throughput: {throughput_half:.2f} tokens/sec, Allocated Memory: {memory_alloc_half} bytes, Reserved Memory: {memory_reserved_half} bytes, Time: {time_half:.2f}s")
