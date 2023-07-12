import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import pandas as pd

def get_model_tokenizer(name):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        load_in_8bit=True,
        device_map={"":0},
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name
    )
    return model, tokenizer

def prepare_model(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model

def get_dataset(path):
    messages = {}
    messages['IDComentario'] = []
    messages['text'] = [] 
    
    gen_text = lambda x: f"### human: Você é um especialista em responder comentários negativos de um cliente a um restaurante. \
                Sua tarefa é responder respeitosamente um comentário negativo de um cliente ao seu restaurante. \
                Dado o comentário do cliente entre, escreva em Português um comentário de resposta de forma respeitosa, \
                empática e não genérica, convencendo o cliente que medidas serão tomadas para resolver o seu problema e \
                que ele poderá voltar a fazer pedidos no restaurante. \
                Certifique-se de usar detalhes específicos do comentário do cliente.\n \
                <{x['comentario']}>\n\
                ### bot: {x['resposta']}"
                
    df_train = pd.read_csv(path)
    for i, linha in df_train.iterrows():
        text = gen_text(linha)
        messages['IDComentario'].append(linha['IDComentario'])
        messages['text'].append(text)
    
    message_df = pd.DataFrame.from_dict(messages)
    return Dataset.from_pandas(message_df)

if __name__ == '__main__':
    model_name = 'tiiuae/falcon-7b'
    path_dataframe = './data/comentarios-respostas-sample.csv'
    
    model, tokenizer = get_model_tokenizer(model_name)
    
    model = prepare_model(model)
    
    config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["query_key_value"],
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    data = get_dataset(path_dataframe)
    
    tokenizer.pad_token = tokenizer.eos_token
    data = data.map(lambda samples: tokenizer(samples["text"], padding=True, truncation=False,), batched=True)

    
    training_args = transformers.TrainingArguments(
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        save_total_limit=4,
        logging_steps=25,
        output_dir="./outputs",
        save_strategy='epoch',
        optim="paged_adamw_8bit",
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.05,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
