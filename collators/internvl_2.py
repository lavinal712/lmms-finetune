import re
from typing import Dict, List, Sequence, Union
import PIL

import torch

from . import register_collator
from .base import BaseDataCollator


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

        for cur_image_paths, system_prompt, cur_convs in zip(image_paths, system_prompts, conversations):
            cur_num_images = len(cur_image_paths)
            cur_image_idx = 0

            cur_input_ids = []
            cur_labels = []

            if system_prompt is not None:
                pass
            
            for i, text in enumerate(cur_convs):
                if i % 2 == 0:
                    num_images = len([m.start() for m in re.finditer("<image>", text)])
                    cur_num_images += num_images

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=None,
        )