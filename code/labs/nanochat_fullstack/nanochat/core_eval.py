"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

_PROMPT_TEMPLATE_MC = Template("""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip())

_PROMPT_TEMPLATE_SCHEMA = Template("""
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip())

_PROMPT_TEMPLATE_LM = Template("""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip())


def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_PROMPT_TEMPLATE_MC.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_PROMPT_TEMPLATE_SCHEMA.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = _PROMPT_TEMPLATE_LM.render(include_continuation=False, **context)
    prompt_with = _PROMPT_TEMPLATE_LM.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    reference = token_sequences[0]
    comparison_sequences = token_sequences[1:]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = reference[idx]
        if not all(seq[idx] == token for seq in comparison_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.inference_mode()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return Bx(T-1) next-token losses and raw logits.
    Callers can argmax only the positions that actually need token predictions.
    """
    batch_size, seq_len = input_ids.size()
    logits = model(input_ids)
    if seq_len <= 1:
        losses = logits.new_empty((batch_size, 0))
    else:
        losses = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].transpose(1, 2),
            input_ids[:, 1:],
            reduction='none',
        )
    return losses, logits


@torch.inference_mode()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, returning a bool-like tensor when possible."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model and get autoregressive losses for each next-token target.
    losses, logits = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = logits[0, si - 1 : ei - 1, :].argmax(dim=-1)
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens)
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        mean_losses = losses.new_empty(len(start_idxs))
        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs)):
            mean_losses[i] = losses[i, si - 1 : ei - 1].mean()
        pred_idx = mean_losses.argmin()
        is_correct = pred_idx == int(item['gold'])
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        if isinstance(is_correct, torch.Tensor):
            correct[idx] = is_correct.to(device=device, dtype=correct.dtype)
        else:
            correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = float(correct.mean().detach().cpu())
    return mean_correct
