import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


def process_input_for_ppl(
        image,
        message,
        pre_message,
        tokenizer,
        transform,
        query_num=64,
        slice_config=None,
):
    data_list, img_list = build_prompt(message, image, tokenizer, slice_config, query_num)
    pre_text_list, _ = build_prompt(pre_message, image, tokenizer, slice_config, query_num)

    bs = len(data_list)
    if img_list == None:
        img_list = [[] for i in range(bs)]
    assert bs == len(img_list)

    input_tensors = [convert_to_tensors(tokenizer, data) for data in data_list]
    model_inputs = {}
    model_inputs["input_ids"] = pad(input_tensors, "input_ids", padding_side="left")
    model_inputs["image_bound"] = [i["image_bound"] for i in input_tensors]

    pixel_values = []
    for i in range(bs):
        img_inps = []
        for img in img_list[i]:
            img_inps.append(transform(img))
        if img_inps:
            pixel_values.append(img_inps)
        else:
            pixel_values.append([])
    model_inputs["pixel_values"] = pixel_values

    model_inputs['position_ids'] = torch.arange(model_inputs['input_ids'].shape[1]).unsqueeze(0).long()
    model_inputs['attention_mask'] = torch.ones_like(model_inputs["input_ids"], dtype=torch.bool)

    labels = model_inputs["input_ids"].clone().long()
    pretext_tensors = [convert_to_tensors(tokenizer, data) for data in pre_text_list]
    pretext_tensors = pad(pretext_tensors, "input_ids", padding_side="left")
    labels[:, :pretext_tensors.shape[1]] = -100

    model_inputs['labels'] = labels

    return model_inputs


def build_prompt(msgs, image, tokenizer, slice_config, query_num):
    prompt = ""
    assert len(msgs) > 0, "must have message"
    for i, msg in enumerate(msgs):
        role = msg["role"]
        content = msg["content"]
        assert role in ["user", "assistant"]
        if i == 0:
            assert role == "user", "The role of first msg should be user"
            if slice_config is not None:
                images, final_placeholder = get_slice_image_placeholder(
                    image, tokenizer, slice_config
                )
                content = final_placeholder + "\n" + content
            else:
                images = [image]
                content = (
                        tokenizer.im_start
                        + tokenizer.unk_token * query_num
                        + tokenizer.im_end
                        + "\n"
                        + content
                )
        prompt += "<用户>" if role == "user" else "<AI>"
        prompt += content
    if role == "user":
        prompt += "<AI>"
    return [prompt], [images]


def get_slice_image_placeholder(image, tokenizer, slice_config):
    image_placeholder = (
            tokenizer.im_start
            + tokenizer.unk_token * slice_config.query_num
            + tokenizer.im_end
    )

    slice_images = []

    source_image, patches, best_grid = slice_image(
        image,
        slice_config.max_slice_nums,
        slice_config.scale_resolution,
        slice_config.patch_size,
    )

    slice_images.append(source_image)
    final_placeholder = image_placeholder

    if len(patches) > 0:
        for i in range(len(patches)):
            for j in range(len(patches[0])):
                slice_images.append(patches[i][j])

        final_placeholder += get_grid_placeholder(
            tokenizer, best_grid, slice_config.query_num
        )

    return slice_images, final_placeholder


def convert_to_tensors(
        tokenizer, input_str, max_inp_length: Optional[int] = 2048
):
    if tokenizer.add_bos_token:
        input_ids = tokenizer.encode(input_str)
    else:
        input_ids = [tokenizer.bos_id] + tokenizer.encode(input_str)
    if max_inp_length is not None:
        input_ids = input_ids[:max_inp_length]
    input_ids = torch.tensor(input_ids, dtype=torch.int32)

    image_start_tokens = torch.where(input_ids == tokenizer.im_start_id)[0]
    # 跳过 im_start
    image_start_tokens += 1
    image_end_tokens = torch.where(input_ids == tokenizer.im_end_id)[0]
    valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
    image_bound = torch.hstack(
        [
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ]
    )

    model_input = {}
    model_input["input_ids"] = input_ids.unsqueeze(0)
    model_input["image_bound"] = image_bound

    return model_input


def pad(orig_items, key, max_length=None, padding_value=0, padding_side="left"):
    items = []
    if isinstance(orig_items[0][key], list):
        assert isinstance(orig_items[0][key][0], torch.Tensor)
        for it in orig_items:
            for tr in it[key]:
                items.append({key: tr})
    else:
        assert isinstance(orig_items[0][key], torch.Tensor)
        items = orig_items

    batch_size = len(items)
    shape = items[0][key].shape
    dim = len(shape)
    assert dim <= 3
    if max_length is None:
        max_length = 0
    max_length = max(max_length, max(item[key].shape[-1] for item in items))
    min_length = min(item[key].shape[-1] for item in items)
    dtype = items[0][key].dtype

    if dim == 1:
        return torch.cat([item[key] for item in items], dim=0)
    elif dim == 2:
        if max_length == min_length:
            return torch.cat([item[key] for item in items], dim=0)
        tensor = torch.zeros((batch_size, max_length), dtype=dtype) + padding_value
    else:
        tensor = (
            torch.zeros((batch_size, max_length, shape[-1]), dtype=dtype)
            + padding_value
        )

    for i, item in enumerate(items):
        if dim == 2:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0])] = item[key][0].clone()
        elif dim == 3:
            if padding_side == "left":
                tensor[i, -len(item[key][0]) :, :] = item[key][0].clone()
            else:
                tensor[i, : len(item[key][0]), :] = item[key][0].clone()

    return tensor




def process_input_dict_for_ppl(
        image,
        conversation,
        tokenizer,
        transform,
        query_num=64,
        patch_size=14,
        slice_config=None,
        batch_vision=None,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    assert conversation[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )
    if slice_config:
        images = []
        source_image, patches, best_grid = slice_image(
            image,
            slice_config["max_slice_nums"],
            slice_config["scale_resolution"],
            slice_config["patch_size"],
        )
        images.append(source_image)
        image_placeholder = default_image_placeholder
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    images.append(patches[i][j])

            image_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_num)
        images = [transform(i) for i in images]
    else:
        images = [transform(image)]
        image_placeholder = default_image_placeholder
    if "<image>" in conversation[0]["content"]:
        conversation[0]["content"] = conversation[0]["content"].replace(
            "<image>", image_placeholder
        )
    else:
        conversation[0]["content"] = (
                image_placeholder + "\n" + conversation[0]["content"]
        )

    input_dict = conversation_to_ids(conversation, tokenizer, llm_type='minicpm')

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes
    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    ret = dict(
        input_ids=input_dict["input_ids"],
        position_ids=input_dict["position_ids"],
        labels=input_dict["target"],
        attention_mask=torch.ones_like(input_dict["input_ids"], dtype=torch.bool),
        pixel_values=[input_dict["pixel_values"]],
        tgt_sizes=input_dict["tgt_sizes"],
        image_bound=input_dict["image_bound"],
    )

    return ret


def conversation_to_ids(conversation, tokenizer, llm_type=None):
    """
    for single image multi-turn conversation
    conversation: [{'role': 'user', 'content': 'Describe this image'},
                   {'role': 'assistant', 'content': 'This is a cat.'}]
    """
    if llm_type == "llama3":
        input_ids, context, raw_msg = conversation_to_ids_llama3(
            conversation, tokenizer
        )
    else:
        input_ids, context, raw_msg = conversation_to_ids_minicpm(
            conversation, tokenizer
        )

    ids = torch.from_numpy(np.hstack(input_ids, dtype=np.int32))
    context = torch.from_numpy(np.hstack(context, dtype=np.int8))

    # build target
    target = torch.full_like(ids, -100, dtype=torch.int32)
    for i in range(1, len(ids)):
        if context[i] == 0:
            target[i - 1] = ids[i]
        if context[i] == 1 and context[i - 1] == 0:
            if hasattr(tokenizer, "eot_id"):
                target[i - 1] = tokenizer.eot_id
            else:
                target[i - 1] = tokenizer.eos_id

    # build image bound
    image_start_tokens = torch.where(ids == tokenizer.im_start_id)[0]
    image_start_tokens += 1
    image_end_tokens = torch.where(ids == tokenizer.im_end_id)[0]
    if len(image_start_tokens) != len(image_end_tokens):
        print("image start token != image end tokens")

    valid_image_nums = max(len(image_start_tokens), len(image_end_tokens))
    image_bound = torch.hstack(
        [
            image_start_tokens[:valid_image_nums].unsqueeze(-1),
            image_end_tokens[:valid_image_nums].unsqueeze(-1),
        ]
    )

    # if len(image_start_tokens) > 0:
    #     image_bound = torch.hstack(
    #         [image_start_tokens.unsqueeze(-1), image_end_tokens.unsqueeze(-1)]
    #     )
    # else:
    #     image_bound = []

    position_ids = torch.arange(ids.size(0)).long()

    return {
        "input_ids": ids,
        "target": target,
        "image_bound": image_bound,
        "raw_msg": raw_msg,
        "position_ids": position_ids
    }


def conversation_to_ids_minicpm(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    for idx, msg in enumerate(conversation):
        role = msg["role"]
        message = msg["content"]
        assert role in ["user", "assistant"]
        if role == "user":
            prefix = "<用户>"
        else:
            prefix = "<AI>"
        # append eos
        if idx == len(conversation) - 1:
            message = message + tokenizer.eos_token
        prefix_ids = tokenizer.encode(prefix)[1:]  # remove bos
        message_ids = tokenizer.encode(message)[1:]

        input_ids.append(prefix_ids)
        input_ids.append(message_ids)

        context.append(np.ones((len(prefix_ids),), dtype=np.int8))
        if role == "assistant":
            context.append(np.zeros((len(message_ids),), dtype=np.int8))
        else:
            context.append(np.ones((len(message_ids),), dtype=np.int8))

        raw_msg += prefix + message

    return input_ids, context, raw_msg


def conversation_to_ids_llama3(conversation, tokenizer):
    raw_msg = ""
    input_ids = []
    context = []
    raw_msg = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False
    )
    input_ids = np.array(input_ids)

    start_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    )[0]
    assistant_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("assistant")
    )[0]
    end_header_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    )[0]
    eot_idxs = np.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|eot_id|>"))[0]

    context = np.ones_like(input_ids, dtype=np.int8)

    for assistant_idx in assistant_idxs:
        if assistant_idx in set((start_header_idxs + end_header_idxs) / 2):
            st = assistant_idx + 3  # assistant<|end_header_id|>\n\n
            for eot_idx in eot_idxs:
                if eot_idx > st:
                    context[st: eot_idx + 1] = 0
                    break

    input_ids = np.hstack(input_ids)
    context = np.hstack(context)

    return input_ids, context, raw_msg



def preprocess(
        image,
        conversation,
        tokenizer,
        transform,
        query_nums=64,
        slice_config=None,
        llm_type=None,
        patch_size=14,
        batch_vision=False,
):
    """
    single image preprocess, the image will be placed at the top of the conversation
    """
    conversation = copy.deepcopy(conversation)
    assert len(conversation) > 1, "conversation length must large than 2"
    assert conversation[0]["role"] == "user", "the first role must be user"

    if slice_config is not None:
        assert isinstance(slice_config, Dict)
        assert "patch_size" in slice_config
        assert "max_slice_nums" in slice_config
        assert "scale_resolution" in slice_config
    default_image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_nums + tokenizer.im_end
    )
    if slice_config:
        images = []
        source_image, patches, best_grid = slice_image(
            image,
            slice_config["max_slice_nums"],
            slice_config["scale_resolution"],
            slice_config["patch_size"],
        )
        images.append(source_image)
        image_placeholder = default_image_placeholder
        if len(patches) > 0:
            for i in range(len(patches)):
                for j in range(len(patches[0])):
                    images.append(patches[i][j])

            image_placeholder += get_grid_placeholder(
                tokenizer, best_grid, query_nums)
        images = [transform(i) for i in images]
    else:
        images = [transform(image)]
        image_placeholder = default_image_placeholder
    if "<image>" in conversation[0]["content"]:
        conversation[0]["content"] = conversation[0]["content"].replace(
            "<image>", image_placeholder
        )
    else:
        conversation[0]["content"] = (
                image_placeholder + "\n" + conversation[0]["content"]
        )

    input_dict = conversation_to_ids(conversation, tokenizer, llm_type)

    if batch_vision:
        tgt_sizes = []
        reshape_images = []
        for image in images:
            H, W = image.shape[1:]
            reshape_image = reshape_by_patch(image, patch_size)
            reshape_images.append(reshape_image)
            tgt_sizes.append([H // patch_size, W // patch_size])
        if tgt_sizes:
            tgt_sizes = torch.Tensor(tgt_sizes).type(torch.int32)

        input_dict["pixel_values"] = reshape_images
        input_dict["tgt_sizes"] = tgt_sizes

    else:
        input_dict["pixel_values"] = images
        input_dict["tgt_sizes"] = []

    return input_dict


def slice_image(
        image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False
):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / \
            (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    source_image = None
    best_grid = None
    patches = []

    if multiple <= 1 or never_split:
        # dont need to slice, upsample
        best_size = find_best_resize(
            original_size, scale_resolution, patch_size, allow_upscale=True
        )
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
    else:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)

        # source image, down-sampling and ensure divided by patch_size
        best_resize = find_best_resize(
            original_size, scale_resolution, patch_size)
        source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)
        candidate_grids = []

        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1

        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error

        refine_size = get_refine_size(
            original_size, best_grid, scale_resolution, patch_size, allow_upscale=True
        )

        refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
        patches = split_to_patches(refine_image, best_grid)

    return source_image, patches, best_grid


def ensure_divide(length, patch_size):
    return max(round(length / patch_size) * patch_size, patch_size)


def find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=False):
    width, height = original_size
    if (width * height > scale_resolution * scale_resolution) or allow_upscale:
        r = width / height
        height = int(scale_resolution / math.sqrt(r))
        width = int(height * r)
    best_width = ensure_divide(width, patch_size)
    best_height = ensure_divide(height, patch_size)
    return (best_width, best_height)


def get_refine_size(
        original_size, grid, scale_resolution, patch_size, allow_upscale=False
):
    width, height = original_size
    grid_x, grid_y = grid

    refine_width = ensure_divide(width, grid_x)
    refine_height = ensure_divide(height, grid_y)

    grid_width = refine_width / grid_x
    grid_height = refine_height / grid_y

    best_grid_size = find_best_resize(
        (grid_width, grid_height),
        scale_resolution,
        patch_size,
        allow_upscale=allow_upscale,
    )

    refine_size = (best_grid_size[0] * grid_x, best_grid_size[1] * grid_y)

    return refine_size


def split_to_patches(image, grid):
    patches = []
    width, height = image.size
    grid_x = int(width / grid[0])
    grid_y = int(height / grid[1])

    for i in range(0, height, grid_y):
        images = []
        for j in range(0, width, grid_x):
            box = (j, i, j + grid_x, i + grid_y)
            patch = image.crop(box)
            images.append(patch)
        patches.append(images)

    return patches


def get_grid_placeholder(tokenizer, grid, query_num):
    image_placeholder = (
            tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    )

    cols = grid[0]
    rows = grid[1]
    slices = []
    for i in range(rows):
        lines = []
        for j in range(cols):
            lines.append(image_placeholder)
        slices.append("".join(lines))
    slice_placeholder = tokenizer.slice_start + \
                        "\n".join(slices) + tokenizer.slice_end
    return slice_placeholder


def reshape_by_patch(image_tensor, patch_size):
    """
    :param image_tensor: shape [3, H, W]
    :param patch_size:
    :return: [3, patch_size, HW/patch_size]
    """
    patches = torch.nn.functional.unfold(
        image_tensor, (patch_size, patch_size), stride=(patch_size, patch_size)
    )

    patches = patches.reshape(image_tensor.size(0), patch_size, patch_size, -1)
    patches = patches.permute(0, 1, 3, 2).reshape(
        image_tensor.size(0), patch_size, -1)
    return patches