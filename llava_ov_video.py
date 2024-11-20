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

warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "/home/panwen.hu/workspace/haokun.lin/checkpoints/llava-onevision-qwen2-0.5b-ov"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)


# def load_video(folder_path, max_frames_num):
#     """
#     Load and uniformly sample frames from a folder containing image files.

#     Args:
#         folder_path (str): Path to the folder containing the frame images.
#         max_frames_num (int): Maximum number of frames to sample.

#     Returns:
#         np.ndarray: Array of sampled frames (frames, height, width, channels).
#     """
#     # Get all files in the folder and sort them (assuming filenames represent frame order)
#     frame_files = sorted(
#         [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
#     )
    
#     # Total number of available frames
#     total_frame_num = len(frame_files)
#     if total_frame_num == 0:
#         raise ValueError("No image files found in the specified folder.")
    
#     # Uniformly sample frame indices
#     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
#     sampled_frame_files = [frame_files[idx] for idx in uniform_sampled_frames]
    
#     # Load frames as numpy arrays
#     spare_frames = []
#     for frame_file in sampled_frame_files:
#         with Image.open(frame_file) as img:
#             spare_frames.append(np.asarray(img))
    
#     return np.array(spare_frames)  # (frames, height, width, channels)


# Load and process video
video_path = "/home/panwen.hu/workspace/haokun.lin/code/LLaVA-NeXT/docs/jobs.mp4"
video_frames = load_video(video_path, 16)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
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
print(text_outputs[0])