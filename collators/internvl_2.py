import re
from typing import Dict, List, Sequence, Union
import PIL

import torch

from . import register_collator
from .base import BaseDataCollator


SYSTEM_MESSAGE = "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"
DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IGNORE_INDEX = -100


@register_collator("internvl-2")
class InternVL2DataCollator(BaseDataCollator):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # images
        # for qwen-vl these are filepaths to the images
        image_paths: List[List] = [instance["images"] for instance in instances]

        # texts
        # the dataset implementation assume conversations are [user, assistant, user, assistant, ...]
        system_prompts: List[Union[str, None]] = [instance["system_prompt"] for instance in instances]
        conversations: List[List] = [instance["conversations"] for instance in instances]
        input_ids = []
        labels = []

        im_start: int = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        im_end: int = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        nl_tokens: List = self.tokenizer("\n", add_special_tokens=False).input_ids
        _system: List = self.tokenizer("system", add_special_tokens=False).input_ids + nl_tokens
        max_len = self.tokenizer.model_max_length

        for cur_image_paths, system_prompt, cur_convs in zip(image_paths, system_prompts, conversations):
            cur_num_images = len(cur_image_paths)
            cur_image_idx = 0

            cur_input_ids = []
            cur_labels = []

            if system_prompt is None:
                system_prompt = SYSTEM_MESSAGE

            system = [im_start] + _system + self.tokenizer(system_prompt, add_special_tokens=False).input_ids + [im_end] + nl_tokens
            cur_input_ids.extend(system)
            cur_labels.extend([im_start] + [self.IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens)
            assert len(cur_input_ids) == len(cur_labels), "Input and label shapes do not match"

            for i, text in enumerate(cur_convs):
                # deal with the special image token format of qwen-vl
                image_token_start_locations = [m.start() for m in re.finditer('<image>', text)]
                for image_token_start in image_token_start_locations:
                    assert cur_image_idx < cur_num_images, "Image index out of bounds"
                    text = text[:image_token_start] + \
                        f"{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 256}{IMG_END_TOKEN}\n" + \
                        text[image_token_start + len("<image>"):]
                    cur_image_idx += 1

                role = "<|im_start|>user" if i % 2 == 0 else "<|im_start|>assistant"
                _input_id = self.tokenizer(role, add_special_tokens=False).input_ids + nl_tokens + \
                    self.tokenizer(text, add_special_tokens=False).input_ids + [im_end] + nl_tokens
                cur_input_ids.extend(_input_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.PAD_TOKEN_ID),
        )