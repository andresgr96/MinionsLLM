from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only, get_chat_template
from datasets import Dataset
import json
import torch 
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from functools import partial


class FineTuningUtils(): 

    def __init__(self, json_data_path: str): 
        self.json_data_path = json_data_path

        # read the json data 
        with open(self.json_data_path, 'r') as file:
            self.json_data = json.load(file)

        # 4bit pre quantized models we support for 4x faster downloading + no OOMs
        self.fourbit_models = [
            "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
            "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
            "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
            "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
            "unsloth/Phi-3-medium-4k-instruct",
            "unsloth/gemma-2-9b-bnb-4bit",
            "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
            "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
            "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "unsloth/Llama-3.2-3B-bnb-4bit",
            "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  # NEW! Llama 3.3 70B!
        ]

    @staticmethod
    def preprocess_json_data_for_unsloth(json_data) -> list:
        """process the json data creating the three pairs of data points for each json blob representing a tree, preprocessing for formatting as a datasets object"""

        # create output packets
        output_packet: list = []

        # extract the tasks and tree from the JSON data
        for packet in json_data:  
            tree = packet["tree"]
            spoon_task = packet["spoon_task"]
            technical_task = packet["technical_task"]
            layman_task = packet["layman_task"]

            # create three sets of pairs
            pairs = [
                {"tree": tree, "task": spoon_task},
                {"tree": tree, "task": technical_task}, 
                {"tree": tree, "task": layman_task}, 
            ]
            for pair in pairs: 
                conversation_data_point = [
                    {"from": "human", "value": pair["task"]},
                    {"from": "gpt", "value": pair["tree"]},
                ]
                
                # define the packet for the output 
                output_packet.append({
                    "conversations": conversation_data_point,
                    "source": "synthetic",
                })

        return output_packet

    def create_dataset_object(self) -> Dataset:
        """Creates a Dataset object from preprocessed JSON data"""
        dataset_pairs = self.preprocess_json_data_for_unsloth(self.json_data)
        return Dataset.from_list(dataset_pairs)

    def formatting_prompts_func(self, tokenizer, examples) -> dict:
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }

    def get_gpu_memory_stats(self, trainer_stats: SFTTrainer, start_gpu_memory: float, max_memory: float) -> None:
        """Get current GPU memory statistics"""
        #@title Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")    


class ModelTrainer(FineTuningUtils):
    
    """class for fine-tuning the model"""

    def __init__(self, json_data_path: str, max_seq_length=2048, dtype=None, load_in_4bit=True):
        super().__init__(json_data_path)
        self.max_seq_length = max_seq_length  # Auto support RoPE Scaling internally
        self.dtype = dtype  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = load_in_4bit  # Use 4bit quantization to reduce memory usage
        
    def load_model(self, model_name="unsloth/Llama-3.2-3B-Instruct"):
        """
        Load the model and tokenizer with specified parameters
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            tuple: (model, tokenizer)
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        return model, tokenizer
    
    def get_peft_model(self):
        """
        Get the PEFT model with LoRA configuration
        
        Args:
            model: The base model to apply PEFT/LoRA to
            
        Returns:
            model: The PEFT-configured model
        """

        # get the model 
        model, tokenizer = self.load_model()

        fast_model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",     # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,   # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
        return fast_model, tokenizer

    def perform_training(self):
        """
        Perform training on the model
        """

        # define the model and tokenizer
        model, tokenizer = self.get_peft_model()

        # create the dataset 
        dataset = self.create_dataset_object()

        # standardize the dataset   
        dataset_modified = standardize_sharegpt(dataset)

        # create partial function with tokenizer parameter set
        formatting_func = partial(self.formatting_prompts_func, tokenizer)

        # format the dataset using partial function
        dataset_modified = dataset_modified.map(formatting_func, batched=True)

        # create the trainer
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset_modified,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                # num_train_epochs = 1, # Set this for 1 full training run.
                max_steps = 60,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
                report_to = "none", # Use this for WandB etc
            ),
        )

        # train the model on the assistant responses only and ignore the loss on the user's inputs
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        # get the gpu memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)  

        # train the model
        training_stats = trainer.train()

        # get the gpu memory stats
        self.get_gpu_memory_stats(training_stats, start_gpu_memory, max_memory)

        return model, tokenizer

    def save_model_gguf(self, model, tokenizer, 
                       hub_repo: str = "martinmashy/finetuned_trees", 
                       quantization_methods: list = ["q4_k_m", "q8_0", "q5_k_m"],
                       token: str = None) -> None:
        """
        Save the fine-tuned model to Hugging Face Hub in GGUF format with multiple quantization options
        
        Args:
            hub_repo (str): Hugging Face Hub repository name (username/repo_name)
            quantization_methods (list): List of quantization methods to use
            token (str, optional): Hugging Face authentication token
            
        Returns:
            None
        """
        # Push model to Hub in GGUF format with specified quantization methods
        model.push_to_hub_gguf(
            hub_repo,
            tokenizer,
            quantization_method=quantization_methods,
            token=token
        )
        
        print(f"Model successfully pushed to {hub_repo} in GGUF format with quantizations: {', '.join(quantization_methods)}")


class ModelInference(ModelTrainer): 

    def __init__(self, model_name: str, max_seq_length: int, dtype: str, load_in_4bit: bool, load_model_inference: bool = False):
        super().__init__(max_seq_length, dtype, load_in_4bit)
        self.model_name = model_name

        if load_model_inference: 
            self.model, self.tokenizer = self.load_model_for_inference()

    def load_model_for_inference(self):
        """
        Load the model for inference
        """
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)

        # define the tokenizer
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )

        # return the model and tokenizer    
        return model, tokenizer

    def run_inference(self, prompt: str, max_new_tokens: int = 64, temperature: float = 1.5, min_p: float = 0.1) -> str:
        """
        Run inference on the model
        
        Args:
            prompt (str): The input prompt for the model
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for sampling
            min_p (float): Minimum probability for nucleus sampling
            
        Returns:
            str: The generated response
        """
   
        # Prepare the messages
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        # Format inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")
        
        # Generate response
        outputs = self.model.generate(
            input_ids=inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            temperature=temperature, 
            min_p=min_p
        )
        
        # Decode and return the response
        return self.tokenizer.batch_decode(outputs)[0]


# Usage example:
if __name__ == "__main__":
    trainer = ModelTrainer('/content/unfiltered_dataset_a.json')
    model, tokenizer = trainer.perform_training()
    trainer.save_model_gguf(model, tokenizer, hub_repo="martinmashy/finetuned_trees", token='')

    # inference example
    #print(trainer.run_inference("What is the capital of France?"))