import argparse

import torch
import os
import time

from fastchat.model import load_model, get_conversation_template, add_model_args
from dataset import CaptionDataset
from torch.utils.data import DataLoader


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

    from IPython import embed
    embed()

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
            f"Input: These are captions of the frames in temporal order within the same video: {pt}. please summarize the whole video according to the frame captions in short. Always answer in one sentence. Output: This video shows"
            for pt in prompts
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
        print(outputs)
        from IPython import embed
        embed()

    # Build the prompt with a conversation template
    # msg = args.message
    # conv = get_conversation_template(args.model_path)
    # conv.append_message(conv.roles[0], msg)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()

    # Run inference
    # inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
    # output_ids = model.generate(
    #     **inputs,
    #     do_sample=True if args.temperature > 1e-5 else False,
    #     temperature=args.temperature,
    #     repetition_penalty=args.repetition_penalty,
    #     max_new_tokens=args.max_new_tokens,
    # )
    #
    # if model.config.is_encoder_decoder:
    #     output_ids = output_ids[0]
    # else:
    #     output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    # outputs = tokenizer.decode(
    #     output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    # )
    #
    # # Print results
    # print(f"{conv.roles[0]}: {msg}")
    # print(f"{conv.roles[1]}: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
