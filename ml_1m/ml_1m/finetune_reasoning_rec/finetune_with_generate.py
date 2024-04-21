import os
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from sklearn.metrics import roc_auc_score

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf", ### "meta-llama/Meta-Llama-3-8B-Instruct", #"baffo32/decapoda-research-llama-7B-hf",  # the only required argument
    train_data_path: str = "./final_data/movie/train.json",
    val_data_path: str = "./final_data/movie/valid.json",
    output_dir: str = "./lora_llama2_chat/sample_8_test",
    sample: int = 16,
    seed: int = 0,
    # training hyperparams
    batch_size: int = 2,
    micro_batch_size: int = 1, ### set to batch_size//2
    num_epochs: int = 2,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2048,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = "",  # either training checkpoint or final adapter

):
    print(
        f"Training lora-llama2-chat model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = 'auto'
    max_memory_mapping = {0: "23GiB", 1: "23GiB"}
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or "auto")}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    tokenizer = LlamaTokenizer.from_pretrained(base_model,
                                               padding_side = "left",
                                               add_eos_token = True,
                                               add_bos_token = True,
                                               token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
                                            )

    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
    # tokenizer.padding_side = "left"  # Allow batched inference
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory = max_memory_mapping,
        token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
    ).eval()
    model.resize_token_embeddings(len(tokenizer),pad_to_multiple_of=8)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        # if (
        #     result["input_ids"][-1] != tokenizer.eos_token_id
        #     and len(result["input_ids"]) < cutoff_len
        #     and add_eos_token
        # ):
        #     result["input_ids"].append(tokenizer.eos_token_id)
        #     result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        # print(result["labels"])

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    

    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    
    # train_data = train_data.shuffle(seed=42)[:sample] if sample > -1 else train_data
    # print(len(train_data))
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    # print(train_data)
    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    print(train_data)
    # print(train_data[0])
    # print("train_data[0]:", train_data[0])
    # print(tokenizer.batch_decode(train_data[0]['labels'], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    
    # train_data = train_data.remove_columns(train_data["train"].column_names)
    # print(train_data)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    def compute_metrics(eval_preds):
        # print(eval_preds[0], eval_preds[1], eval_preds[2])
        pre, labels = eval_preds
        # print("Lengths:", len(labels), len(pre), len(pre[0]), len(pre[1]))
        # print("Dimensions:", len(labels[0]))
        # print("labels:", labels)
        # print(pre[1], pre[0])
        auc = roc_auc_score(pre[1], pre[0])
        return {'auc': auc}
    
    def preprocess_logits_for_metrics(logits, labels):
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        print(f"len of logits: {logits.shape}")
        # print(f"logits dimension: {len(logits[0])}")
        # print(f"len of labels: {labels.shape} --- labels: {labels}")
        # print(f"labels dimension: {len(labels[0])}")
        # gts = torch.tensor([[num if num!=-100 else 32000 for num in label] for label in labels.tolist()])
        gts = []
        for label in labels.tolist():
            temp = []
            for num in label:
                if num!=-100:
                    temp.append(num)
                else:
                    temp.append(2)
            gts.append(temp)
        gts = torch.tensor(gts)

        # print("labels:", labels)
        # print("gts:", gts)
        gt_texts = tokenizer.batch_decode(gts, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        print("GT:", len(gt_texts), gt_texts)

        ### Generate output from model - 
        gt_inputs = [text.split("[/INST]")[0]+"[/INST]" for text in gt_texts]
        gt_labels = [text.split("[/INST]")[1] for text in gt_texts]
        print("="*100)
        print("GT Inputs:", len(gt_inputs), gt_inputs)
        print("="*100)
        print("GT Labels:", len(gt_labels), gt_labels)

        intermediate_inputs = tokenizer(
            gt_inputs,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        intermediate_generated_ids = model.generate(
                                    intermediate_inputs.input_ids
                                    )
        intermediate_outputs = tokenizer.batch_decode(intermediate_generated_ids)
        print("*"*100)
        print("Intermediate Outputs:", len(intermediate_outputs), intermediate_outputs)

        labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
        labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits.softmax(dim=-1)
        argmax_indices = torch.argmax(logits, dim=-1)
        # print("Predicted Indices:", argmax_indices)
        print("-"*100)
        print("Predicted:", tokenizer.batch_decode(argmax_indices, skip_special_tokens=False, clean_up_tokenization_spaces=True))
        logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
        print(logits)
        print(logits[:, 1][2::3], gold[2::3])
        return logits[:, 1][2::3], gold[2::3]

    os.environ["WANDB_DISABLED"] = "true"
    
    if sample > -1:
        if sample <= 128 :
            eval_step = 4
        else:
            eval_step = sample / 128 * 2
    # print("sample: ", sample)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=20,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            # report_to="wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
            # eval_accumulation_steps=10,
        ),
        # data_collator = transformers.DataCollatorWithPadding(
        #     tokenizer, pad_to_multiple_of = 8, return_tensors="pt", padding = True,
        #     ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator=transformers.DataCollatorForLanguageModeling(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False,
        # ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    ### Using SFTTrainer
    # def formatting_prompts_func(example):
    #     output_texts = []
    #     for i in range(len(example['instruction'])):
    #         text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Response: {example['output'][i]}"
    #         output_texts.append(text)
    #     return output_texts

    # response_template = "### Response:"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)

    # trainer = SFTTrainer(
    #     model,
    #     train_dataset=train_data,
    #     eval_dataset=val_data,
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=micro_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         warmup_steps=20,
    #         num_train_epochs=num_epochs,
    #         learning_rate=learning_rate,
    #         fp16=True,
    #         logging_steps=20,
    #         optim="adamw_torch",
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=eval_step,
    #         save_steps=eval_step,
    #         output_dir=output_dir,
    #         save_total_limit=1,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="eval_auc",
    #         ddp_find_unused_parameters=False if ddp else None,
    #         group_by_length=group_by_length,
    #         report_to=None,
    #         # report_to="wandb" if use_wandb else None,
    #         # run_name=wandb_run_name if use_wandb else None,
    #         # eval_accumulation_steps=10,
    #     ),
    #     # data_collator=transformers.DataCollatorForSeq2Seq(
    #     #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    #     # ),
    #     # data_collator=transformers.DataCollatorForLanguageModeling(
    #     #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", mlm=False,
    #     # ),
    #     max_seq_length = 2048,
    #     formatting_func = formatting_prompts_func,
    #     data_collator = collator,
    #     compute_metrics=compute_metrics,
    #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #     callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    # )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.train()


    model.save_pretrained(output_dir, safe_serialization = False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    return f"""[INST]{data_point["instruction"]}{data_point["input"]}[/INST]{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)