import platform

from consistency.Medalpaca.inferer import Inferer


def init_medalpaca_model(args):
    # --- Flags from the original code ---
    load_in_8bit = False
    cache_dir = None
    
    print(f"Loading model {args.model}...")
    if args.model == "medalpaca-30b":
        base_model = "decapoda-research/llama-30b-hf"
        model_name = "medalpaca/medalpaca-lora-30b-8bit"
        peft = True

    elif args.model == "medalpaca-13b":
        base_model = "decapoda-research/llama-13b-hf"
        model_name = "medalpaca/medalpaca-lora-13b-8bit"
        peft = True

    elif args.model == "medalpaca-7b":

        base_model = "../PPLM/models_hf/7B"
        model_name = "medalpaca/medalpaca-7b"
        model_name = "medalpaca/medalpaca-lora-7b-16bit"
        peft = True

        cache_dir = "../medAlpaca/medalpaca-7b"

    else:
        raise ValueError(f"Unknown model: {args.model}")


    prompt_template = f"consistency/Medalpaca/prompt_templates/medalpaca_consistency.json"

    # ------------------------------------

    # Only initialize this model on a Linux machine, which has sufficient GPU memory.

    print("peft", peft)
    print("load_in_8bit", load_in_8bit)
    if platform.system() == "Linux":
        model = Inferer(
            model_name=model_name,
            prompt_template=prompt_template,
            # f"../medalpaca/prompt_templates/medalpaca.json",
            base_model=base_model,
            peft=peft,
            load_in_8bit=load_in_8bit,
            args=args,
            cache_dir=cache_dir,
        )

    else:
        model = None

    return model