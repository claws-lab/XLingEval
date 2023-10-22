import platform
import sys
import json
from typing import List

import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from .handler import DataHandler

assert torch.cuda.is_available() or platform.system() == "Darwin", "No cuda device detected"


class Inferer:
    """
    A basic inference class for accessing medAlpaca models programmatically.

    This class provides methods for loading supported medAlpaca models, tokenizing inputs,
    and generating outputs based on the specified model and configurations.

    Attributes:
        available_models (dict): A dictionary containing the supported models and their configurations.

    Args:
        model_name (str): The name of the medAlpaca model to use for inference.
        prompt_template (str): The path to the JSON file containing the prompt template.
        base_model (str, optional): If LoRA is used, this should point to the bases model weigts
        model_max_length: (int, optional): Number of input tokens to the model. Default is 512.
        load_in_8bit (bool, optional): Wether a quantized model should be loaded. Default is False
        torch_dtype (torch.dtype, optional): The torch datatype to load the base model. Default is float16
        peft (bool, optional): If the model was trainied in 8bit or with LoRA, PEFT library should be used
            to load the model. Default is False. 

    Example:

        medalpaca = medAlapaca("medalpaca/medalapca-7b", "prompts/alpaca.json")
        response = medalpaca(input="What is Amoxicillin?")
    """
        
    def __init__(
        self, 
        model_name: str, 
        prompt_template: str,
        base_model: str = None,
        model_max_length: int = 512,
        load_in_8bit: bool = False, 
        torch_dtype: torch.dtype = torch.float16, 
        peft: bool = False,
        args = None,
        cache_dir = None,
    ) -> None:
        self.args=args
        if base_model and not peft: 
            raise ValueError(
                "You have specified a base model, but `peft` is false", 
                "This would load the base model only"
            )


        
        tokenizer = self._load_tokenizer(base_model or model_name, peft, cache_dir=cache_dir)
                
        self.data_handler = DataHandler(
            tokenizer,
            prompt_template = prompt_template, 
            model_max_length = model_max_length,
            train_on_inputs = False,

        )
        self.data_handler.prompt_template['primer'].replace("LANGUAGE_NAME", args.target_language)


        self.model = self._load_model(
            model_name=model_name,
            base_model=base_model or model_name,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            peft=peft,
            cache_dir=cache_dir,
        )
        
   
    def _load_model(
        self, 
        model_name: str, 
        base_model: str, 
        load_in_8bit: bool, 
        torch_dtype: torch.dtype, 
        peft: bool,
        cache_dir: str = None,
    ) -> torch.nn.Module:

        if "llama" in base_model.lower(): 
            load_model = LlamaForCausalLM
        else: 
            load_model = AutoModelForCausalLM

        if cache_dir:
            model = load_model.from_pretrained(
                cache_dir,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                device_map="auto",
                local_files_only=True, use_auth_token=False
            )

            if peft:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    model_id=cache_dir,
                    torch_dtype=torch_dtype,
                    device_map={
                        "": 0
                    },
                )


        else:
            model = load_model.from_pretrained(
                base_model,
                load_in_8bit=load_in_8bit,
                torch_dtype=torch_dtype,
                device_map="auto",
            )

            if peft:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    model_id=model_name,
                    torch_dtype=torch_dtype,
                    device_map={"": 0},
                )
        
        if not load_in_8bit:
            model.half()
            
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        return model
    
    def _load_tokenizer(self, model_name: str, peft: bool, cache_dir: str = None):
        if "llama" in model_name.lower() or not peft:
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        else:
            if cache_dir:
                tokenizer = AutoTokenizer.from_pretrained(cache_dir,
                                                          local_files_only=True, use_auth_token=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"    
        return tokenizer
        
    def __call__(
        self,
        input: str,
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,
    ) -> str:
        """
        Generate a response from the medAlpaca model using the given input and instruction.

        Args:
            input (str):
                The input text to provide to the model.
            instruction (str, optional):
                An optional instruction to guide the model's response.
            output (str, optional): 
                Prepended to the models output, e.g. for 1-shot prompting
            max_new_tokens (int, optional): 
                How many new tokens the model can generate
            verbose (bool, optional): 
                If True, print the prompt before generating a response.
            **generation_kwargs:
                Keyword arguments to passed to the `GenerationConfig`.
                See here for possible arguments: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/text_generation

        Returns:
            str: The generated response from the medAlpaca model.
        """

        prompt = self.data_handler.generate_prompt(instruction = instruction, input = input, output = output)
        if verbose:
            print(prompt)
            
        input_tokens = self.data_handler.tokenizer(prompt, return_tensors="pt")
        input_token_ids = input_tokens["input_ids"].to("cuda")

        generation_config = GenerationConfig(**generation_kwargs)
        

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        generation_output_decoded = self.data_handler.tokenizer.decode(generation_output.sequences[0])
        split = f'{self.data_handler.prompt_template["output"]}{output or ""}'
        response = generation_output_decoded.split(split)[-1].strip()
        return response

    def batch_inference(self, inputs: List[str],
        instruction: str = None,
        output: str = None,
        max_new_tokens: int = 128,
        verbose: bool = False,
        **generation_kwargs,):

        prompts = []

        for input in inputs:
            prompt = self.data_handler.generate_prompt(instruction=instruction,
                                                       input=input,
                                                       output=output)
            if verbose:
                print(prompt)

            prompts += [prompt]


        

        input_tokens = self.data_handler.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_token_ids = input_tokens["input_ids"].to("cuda")


        # print("prompts\n", prompts)
        # print("input_token_ids\n", input_token_ids)
        
        generation_config = GenerationConfig(**generation_kwargs)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_token_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        generation_output_decoded = [self.data_handler.tokenizer.decode(
            generation_output.sequences[i], skip_special_token=True) for i in
            range(
            generation_output.sequences.shape[0])]
        split = f'{self.data_handler.prompt_template["output"]}{output or ""}'

        responses = []
        for i in range(len(generation_output_decoded)):
            response = generation_output_decoded[i].split(split)[-1].strip(

            ).replace("<unk>", "")
            responses += [response]
        return responses
