


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

def build_tokenize(tokenizer,CUTOFF_LEN=256):
    def tokenize(prompt):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )
        return {
            "input_ids": result["input_ids"][:-1],
            "attention_mask": result["attention_mask"][:-1],
        }
    return tokenize

def build_generate_and_tokenize_prompt(tokenizer,CUTOFF_LEN=256):
    def generate_and_tokenize_prompt(data_point):
        # This function masks out the labels for the input,
        # so that our loss is computed only on the response.
        user_prompt = (
            (
                f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    
    ### Instruction:
    {data_point["instruction"]}
    ### Input:
    {data_point["input"]}
    
    ### Response:
    """
            )
            if data_point["input"]
            else (
                f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    
    ### Instruction:
    {data_point["instruction"]}
    
    ### Response:
    """
            )
        )

        len_user_prompt_tokens = (
            len(
                tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=CUTOFF_LEN + 1,
                    # bug repaired, no need pad the prompt -whj
                    padding=False,
                )["input_ids"]
            )
            - 1
        )  # no eos token
        full_tokens = tokenizer(
            user_prompt + data_point["output"],
            truncation=True,
            max_length=CUTOFF_LEN + 1,
            padding="max_length",
        )["input_ids"][:-1]

        return {
            "input_ids": full_tokens,
            "labels": [-100] * len_user_prompt_tokens
            + full_tokens[len_user_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

    return generate_and_tokenize_prompt