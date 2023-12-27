"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import argparse

import torch
import os
import time

from fastchat.model import load_model, add_model_args
from dataset import CaptionDataset
from torch.utils.data import DataLoader


def seconds_to_hms(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)

    return f'{hours:02d} h, {minutes:02d} m, {seconds:02d} s'


@torch.inference_mode()
def main(args):
    os.makedirs(args.output, exist_ok=True)

    # load dataset
    data_set = CaptionDataset(caption_path=args.caption_path)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.batch_size,
                             num_workers=8,
                             pin_memory=True,
                             shuffle=False)

    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug
    )

    # Inference
    total_iters = len(data_loader)
    for cur_iter, (filenames, prompts, captions, actions) in enumerate(data_loader):
        start_time = time.time()

        new_prompts = [
            f"Input: These are captions of the frames in temporal order within the same video: {cp}. please summarize the whole video according to the frame captions in short. Always answer in one sentence. Output: This video shows"
            for cp in captions
        ]

        inputs = tokenizer(new_prompts, padding=True, truncation=True, return_tensors="pt").to(args.device)

        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids_new = output_ids
        else:
            output_ids_new = [output_ids[i][len(inputs["input_ids"][i]):] for i in range(len(output_ids))]
        outputs = tokenizer.batch_decode(
            output_ids_new, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        outputs = [op.strip() for op in outputs]

        if (cur_iter + 1) % args.print_freq == 0:
            batch_time = time.time() - start_time
            time_to_end = batch_time * (total_iters - cur_iter - 1)
            print(
                f'prompt: {new_prompts}\n'
                f'outputs: {outputs}'
            )
            print(
                f'Job: [{args.job_id} / {args.job_num}] Iter: {cur_iter + 1} / {total_iters} ({(cur_iter + 1) / total_iters * 100:.2f}%)'
                f' Batch Time: {batch_time:.2f} TTE: {seconds_to_hms(time_to_end)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # job
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--job-num", type=int, default=1)
    # model
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")
    # dataset
    parser.add_argument("--caption-path", type=str)
    parser.add_argument("--batch-size", type=int, default=8)
    # output
    parser.add_argument("--output", type=str, help="the path to the output dir")
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--print-freq", type=int, default=1)
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
