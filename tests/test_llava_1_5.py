import unittest

import torch
from PIL import Image

from collators import LLaVA1_5_DataCollator
from loaders import LLaVA1_5_ModelLoader


class LLaVA1_5_Test(unittest.TestCase):
    def setUp(self):
        self.model_hf_path = "llava-hf/llava-1.5-7b-hf"
        self.model_local_path = "/aiarena/group/gmgroup/hongyq/models/llava-hf/llava-1.5-7b-hf"
        self.compute_dtype = torch.bfloat16

        self.instances = [
            {
                "images": [Image.open("example_data/images/celeba/000091.jpg").convert("RGB")],
                "videos": [],
                "conversations": [
                    "<image>Describe the person's appearance.",
                    "This person has no smile. This person looks extremely young and has no eyeglasses, and no fringe. He doesn't have any mustache.",
                ],
                "system_prompt": None,
            }
        ]

    def tearDown(self):
        pass
    
    def test_collator(self):
        loader = LLaVA1_5_ModelLoader(
            model_hf_path=self.model_hf_path,
            model_local_path=self.model_local_path,
            compute_dtype=self.compute_dtype,
        )
        _, tokenizer, processor, config = loader.load(load_model=False)
        tokenizer.model_max_length = 2048
        collator = LLaVA1_5_DataCollator(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            mask_question_tokens=True,
        )
        output = collator(self.instances)

    def test_load(self):
        loader = LLaVA1_5_ModelLoader(
            model_hf_path=self.model_hf_path,
            model_local_path=self.model_local_path,
            compute_dtype=self.compute_dtype,
        )
        model, tokenizer, processor, config = loader.load()
