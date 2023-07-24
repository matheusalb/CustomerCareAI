from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import (
    PeftModel
)

def load_tokenizer_and_finetuned_qlora_model_FALCON(
    model_name, 
    finetuned_model_path,
    bnb_config,
    finetuned_tokenizer_path=None
    ):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
    )
    
    if finetuned_tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            finetuned_tokenizer_path,
        )
        model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(model, finetuned_model_path)
    
    return model, tokenizer