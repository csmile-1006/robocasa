import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict

import imageio
import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
from hydra.utils import to_absolute_path
from thop import profile
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, logging


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def torch_save_model(model, model_path, cfg=None, previous_masks=None):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "previous_masks": previous_masks,
        },
        model_path,
    )


def torch_load_model(model_path, map_location=None):
    model_dict = torch.load(model_path, map_location=map_location, weights_only=False)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    if "previous_masks" in model_dict:
        previous_masks = model_dict["previous_masks"]
    return model_dict["state_dict"], cfg, previous_masks


def get_train_test_loader(dataset, train_ratio, train_batch_size, test_batch_size, num_workers=(0, 0)):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers[0],
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers[1],
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def confidence_interval(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def compute_flops(algo, dataset, cfg):
    model = copy.deepcopy(algo.policy)
    tmp_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    data = next(iter(tmp_loader))
    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    macs, params = profile(model, inputs=(data,), verbose=False)
    GFLOPs = macs * 2 / 1e9
    MParams = params / 1e6
    del model
    return GFLOPs, MParams


def create_experiment_dir(cfg):
    prefix = "experiments"
    if cfg.pretrain_model_path != "":
        prefix += "_finetune"
    if cfg.data.task_order_index > 0:
        prefix += f"_permute{cfg.data.task_order_index}"
    if cfg.task_embedding_format == "one-hot":
        prefix += "_onehot"
    if cfg.task_embedding_format == "clip":
        prefix += "_clip"
    if cfg.task_embedding_format == "gpt2":
        prefix += "_gpt2"
    if cfg.task_embedding_format == "roberta":
        prefix += "_roberta"

    experiment_dir = (
        f"./{prefix}/{cfg.benchmark_name}/{cfg.lifelong.algo}/" + f"{cfg.policy.policy_type}_seed{cfg.seed}"
    )

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # look for the most recent run
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1

    experiment_dir += f"/run_{experiment_id:03d}"
    cfg.experiment_dir = experiment_dir
    cfg.experiment_name = "_".join(cfg.experiment_dir.split("/")[2:])
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    return True


def get_task_embs(cfg, descriptions: Dict[str, str]):
    # descriptions: list of task descriptions with duplicates
    # logic:
    # 1. remove duplicates and compute task embeddings
    # 2. apply embeddings to each task description --> make the same set of task descriptions with duplicates
    # 3. return the task embeddings
    logging.set_verbosity_error()

    # 1. remove duplicates and compute task embeddings
    unique_descriptions = [desc for task_descs in descriptions for desc in set(task_descs.values())]
    # 1-1. make a dict of key: demo_idx, value: task_description index in descriptions
    desc_idx_dict = {desc: i for i, desc in enumerate(unique_descriptions)}
    demo_list_to_desc_idx_dict = {}

    for i1, desc in enumerate(descriptions):
        for demo_key, demo_inst in desc.items():
            demo_list_to_desc_idx_dict[f"{i1}.{demo_key}"] = desc_idx_dict[demo_inst]

    if cfg.task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = cfg.task_embedding_one_hot_offset
        descriptions = [f"Task {i + offset}" for i in range(len(descriptions))]

    if cfg.task_embedding_format == "bert" or cfg.task_embedding_format == "one-hot":
        tz = AutoTokenizer.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
        model = AutoModel.from_pretrained("bert-base-cased", cache_dir=to_absolute_path("./bert"))
        tokens = tz(
            text=unique_descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()
    elif cfg.task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=unique_descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif cfg.task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=unique_descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif cfg.task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=unique_descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]

    # 2. apply embeddings to each task description --> make the same set of task descriptions with duplicates
    outputs = []
    for i1, desc in enumerate(descriptions):
        _outputs = {}
        for demo_key in desc.keys():
            _outputs[demo_key] = task_embs[demo_list_to_desc_idx_dict[f"{i1}.{demo_key}"]]
        outputs.append(_outputs)
    return outputs


class Timer:
    def __enter__(self):
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        self.value = (end_time - self.start_time) / (10**9)

    def get_elapsed_time(self):
        return self.value


class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def append_image(self, img, idx=0):
        """Directly append an image to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            self.image_buffer[idx].append(img)

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image"):
        """Append a camera observation to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            if not done:
                self.image_buffer[idx].append(obs[camera_name][::-1])
            else:
                if self.last_images[idx] is None:
                    self.last_images[idx] = obs[camera_name][::-1]
                original_image = np.copy(self.last_images[idx])
                blank_image = np.ones_like(original_image) * 128
                blank_image[:, :, 0] = 0
                blank_image[:, :, -1] = 0
                transparency = 0.7
                original_image = original_image * (1 - transparency) + blank_image * transparency

                self.image_buffer[idx].append(original_image.astype(np.uint8))

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_vector_obs(self, obs, dones, camera_name="agentview_image"):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name)

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, "video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")
