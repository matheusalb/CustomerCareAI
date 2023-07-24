from peft import(
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

from datasets import load_dataset, Dataset

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    return tokenizer


def create_and_prepare_model_FALCON(model_name, peft_config, bnb_config):
    tokenizer = get_tokenizer(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    
    model.resize_token_embeddings(len(tokenizer))
    
    # Memory Effiecency
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model, True)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer




def create_and_prepare_model(model_name, peft_config, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    
    tokenizer = get_tokenizer(model_name)
    
    # Memory Effiecency
    model.gradient_checkpointing_enable()
    
    model = prepare_model_for_kbit_training(model, True)
    model = get_peft_model(model, peft_config)
    
    return model, tokenizer


def get_tokenizer_LLaMA(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(
        model_name
    )
    return tokenizer

def create_and_prepare_LLaMA_model(model_name, peft_config, bnb_config):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    
    model = prepare_model_for_kbit_training(model, True)
    model = get_peft_model(model, peft_config)
    tokenizer = get_tokenizer_LLaMA(model_name)
    
    return model, tokenizer

# model.config.use_cache = False


# tokenizer.pad_token = (
#     0
# )
# tokenizer.padding_side = "left"

