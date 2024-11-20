import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import tqdm

from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default="/home/panwen.hu/workspace/haokun.lin/code/open-eqa/data/open-eqa-v0.json", help="path to EQA dataset (default: data/open-eqa-v0.json)")
    parser.add_argument("--model_path", type=str, default="/home/panwen.hu/workspace/haokun.lin/checkpoints/llava-onevision-qwen2-0.5b-si", help="OpenAI model (default: gpt-4-vision-preview)")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--frames-directory", type=Path, default="/home/panwen.hu/workspace/haokun.lin/data/video/frames", help="path image frames (default: data/frames/)")
    parser.add_argument("--num-frames", type=int,default=50, help="num frames in gpt4v (default: 50)")
    parser.add_argument("--image-size", type=int,default=512, help="image size (default: 512)")
    parser.add_argument("--seed", type=int, default=1234, help="gpt seed (default: 1234)")
    parser.add_argument("--temperature", type=float, default=0.2, help="gpt temperature (default: 0.2)")
    parser.add_argument("--max_tokens", type=int, default=128, help="gpt maximum tokens (default: 128)")
    parser.add_argument("--output-directory", type=Path, default="/home/panwen.hu/workspace/haokun.lin/data/output", help="output directory (default: data/results)")
    parser.add_argument("--force", action="store_true", help="continue running on API errors (default: false)")
    parser.add_argument("--dry-run", action="store_true", help="only process the first 5 questions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    args.output_directory = args.output_directory / Path(model_name)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}.json".format(model_name)
    )
    return args


def load_video(folder_path, frames_num):
    """
    Load and uniformly sample frames from a folder containing image files.

    Args:
        folder_path (str): Path to the folder containing the frame images.
        max_frames_num (int): Maximum number of frames to sample.

    Returns:
        np.ndarray: Array of sampled frames (frames, height, width, channels).
    """
    # Get all files in the folder and sort them (assuming filenames represent frame order)

    frame_files = sorted(folder_path.glob("*-rgb.png"))
    
    # Total number of available frames
    total_frame_num = len(frame_files)
    if total_frame_num == 0:
        raise ValueError("No image files found in the specified folder.")
    
    # Uniformly sample frame indices
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, frames_num, dtype=int)
    sampled_frame_files = [frame_files[idx] for idx in uniform_sampled_frames]
    
    # Load frames as numpy arrays
    spare_frames = []
    for frame_file in sampled_frame_files:
        with Image.open(frame_file) as img:
            spare_frames.append(np.asarray(img))
    
    return np.array(spare_frames)  # (frames, height, width, channels)


def main(args: argparse.Namespace):
    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    llava_model_args = {
    "multimodal": True,
    }
    # load model
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, args.model_name, device_map=args.device_map, attn_implementation="sdpa", **llava_model_args)
    model.eval()

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break
        
        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing
        
        # extract scene paths
        frames_folder = args.frames_directory / item["episode_history"] # /l/users/haokun.lin/data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE
        # file_num = len(os.listdir(frames_folder))
        # video_time = round(file_num / 30, 2)
        # frames = sorted(folder.glob("*-rgb.png"))
        # indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int) 
        # paths = [str(frames[i]) for i in indices] # ['/l/users/haokun.lin/data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00000-rgb.png', '/l/users/haokun.lin/data/frames/hm3d-v0/000-hm3d-BFRyYbPCCPE/00098-rgb.png']
        question = item["question"] # What is the white object on the wall above the TV?

        # Load and process video
        # video_path = "jobs.mp4"
        video_frames = load_video(frames_folder, args.num_frames)
        # print(video_frames.shape) # (16, 1024, 576, 3)
        image_tensors = []
        frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
        image_tensors.append(frames)


        # Prepare conversation input
        conv_template = "qwen_1_5"
        question = f"{DEFAULT_IMAGE_TOKEN}\n" + question

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)
        image_sizes = [frame.size for frame in video_frames]

        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        answer = text_outputs[0]

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))

if __name__ == "__main__":
    main(parse_args())