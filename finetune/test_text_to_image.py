#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import shutil
import sys
import tempfile

from diffusers import DiffusionPipeline, UNet2DConditionModel  # noqa: E402


sys.path.append("..")
from finetune.test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402

import logging.config
import json
FORMATTER = logging.Formatter('"%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(message)s"')

logger = logging.getLogger(__name__) 
sys.stdout.flush()
console_handler = logging.StreamHandler()
console_handler.setFormatter(FORMATTER)
logger.addHandler(console_handler) 
logger.setLevel(logging.DEBUG)

VERSION = "v167_both3"
#./finetune/outputSD2_v163_vision
#stabilityai/stable-diffusion-2-1
class TextToImage(ExamplesTestsAccelerate):
    def test_text_to_image(self):
        output_directory="finetune/outputSD2_" + VERSION
        test_args = f"""
            finetune/train_text_to_image_both.py
            --pretrained_model_name_or_path ./finetune/outputSD2_v164_vision3
            --train_data_dir ../datasets/finetune-dataset/train
            --resolution 256
            --mixed_precision fp16
            --allow_tf32
            --gradient_checkpointing
            --train_batch_size 16 
            --gradient_accumulation_steps 1
            --num_train_epochs 5
            --learning_rate 2e-08
            --lr_scheduler constant
            --lr_warmup_steps 0
            --input_perturbation 0.1
            --center_crop
            --random_flip
            --output_dir {output_directory}
            """.split()

        output = run_command(self._launch_args + test_args, return_stdout=True)
        print(output)
        # save_pretrained smoke test
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "unet", "diffusion_pytorch_model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "scheduler", "scheduler_config.json")))

    def test_text_to_image_checkpointing(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 1
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 2 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # no checkpoint-2 -> check old checkpoints do not exist
            # check new checkpoints exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-5"},
            )

    def test_text_to_image_checkpointing_use_ema(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 2 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=1
                --resume_from_checkpoint=checkpoint-4
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # no checkpoint-2 -> check old checkpoints do not exist
            # check new checkpoints exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-5"},
            )

    def test_text_to_image_checkpointing_checkpoints_total_limit(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 6, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # check checkpoint directories exist
            # checkpoint-2 should have been deleted
            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-4", "checkpoint-6"})

    def test_text_to_image_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"
        prompt = "a prompt"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # resume and we should try to checkpoint at 6, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 8
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --checkpoints_total_limit=2
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(prompt, num_inference_steps=1)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8"},
            )


class TextToImageSDXL(ExamplesTestsAccelerate):
    def test_text_to_image_sdxl(self):
        output_directory="finetune/outputSDXL"
        test_args = f"""
            finetune/train_text_to_image_sdxl.py
            --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0
            --train_data_dir ../datasets/finetune-dataset/train
            --resolution 512
            --gradient_checkpointing
            --mixed_precision fp16
            --center_crop
            --random_flip
            --train_batch_size 1
            --gradient_accumulation_steps 1
            --max_train_steps 2
            --learning_rate 1e-05
            --scale_lr
            --lr_scheduler constant
            --lr_warmup_steps 0
            --output_dir {output_directory}
            """.split()
            

        run_command(self._launch_args + test_args)
        # save_pretrained smoke test
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "unet", "diffusion_pytorch_model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(output_directory, "scheduler", "scheduler_config.json")))
