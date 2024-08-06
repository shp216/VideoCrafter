import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange
from decord import VideoReader, cpu

class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        
        # youtube_id가 '#NAME?'인 경우 건너뛰기
        if video_dict['youtube_id'] == '#NAME?':
            print(f"Skipping video {idx} due to invalid youtube_id")
            return None, None
        
        print(f"Processing video {idx}: {video_dict}")  # 디버깅 메시지 추가
          
        videoid, youtube_id, name = video_dict['audiocap_id'], video_dict['youtube_id'], video_dict['caption']
        
        video_filename = f"audiocaps_{youtube_id}.mp4"
        video_dir = os.path.join(self.video_folder, video_filename)
        
        if not os.path.exists(video_dir):
            print(f"Video file {video_dir} not found.")  # 파일 존재 여부 확인
            return None, None

        video_reader = VideoReader(video_dir, ctx=cpu(0))
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                if pixel_values is None:  # None인 경우 새로운 인덱스를 선택하여 다시 시도
                    idx = random.randint(0, self.length-1)
                    continue
                break
            except Exception as e:
                print(f"Error processing video {idx}: {e}")  # 예외 처리 메시지 추가
                idx = random.randint(0, self.length-1)  # 예외 발생 시 새로운 인덱스를 선택하여 다시 시도

        pixel_values = self.pixel_transforms(pixel_values)
        pixel_values = rearrange(pixel_values, "f c h w -> c f h w")
        sample = dict(pixel_values=pixel_values, text=name)
        return sample





# import os, io, csv, math, random
# import numpy as np
# from einops import rearrange
# from decord import VideoReader, cpu

# import torch
# import torchvision.transforms as transforms
# from torch.utils.data.dataset import Dataset

# class WebVid10M(Dataset):
#     def __init__(
#             self,
#             csv_path, video_folder,
#             sample_size=256, sample_stride=4, sample_n_frames=16,
#             is_image=False,
#         ):
#         print(f"loading annotations from {csv_path} ...")
#         with open(csv_path, 'r') as csvfile:
#             self.dataset = list(csv.DictReader(csvfile))
#         self.length = len(self.dataset)
#         print(f"data scale: {self.length}")

#         self.video_folder = video_folder
#         self.sample_stride = sample_stride
#         self.sample_n_frames = sample_n_frames
#         self.is_image = is_image
        
#         sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
#         self.pixel_transforms = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.Resize(sample_size[0]),
#             transforms.CenterCrop(sample_size),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
#         ])
    
#     def get_batch(self, idx):
#         video_dict = self.dataset[idx]
        
#         # youtube_id가 '#NAME?'인 경우 건너뛰기
#         if video_dict['youtube_id'] == '#NAME?':
#             print(f"Skipping video {idx} due to invalid youtube_id")
#             return None, None
        
#         print(f"Processing video {idx}: {video_dict}")  # 디버깅 메시지 추가
          
#         videoid, youtube_id, name = video_dict['audiocap_id'], video_dict['youtube_id'], video_dict['caption']
        
#         video_filename = f"audiocaps_{youtube_id}.mp4"
#         video_dir = os.path.join(self.video_folder, video_filename)
        
#         if not os.path.exists(video_dir):
#             print(f"Video file {video_dir} not found.")  # 파일 존재 여부 확인
#             return None, None

#         video_reader = VideoReader(video_dir, ctx=cpu(0))
#         video_length = len(video_reader)
        
#         if not self.is_image:
#             clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
#             start_idx = random.randint(0, video_length - clip_length)
#             batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
#         else:
#             batch_index = [random.randint(0, video_length - 1)]

#         pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
#         pixel_values = pixel_values / 255.
#         del video_reader

#         if self.is_image:
#             pixel_values = pixel_values[0]
        
#         return pixel_values, name
#         # video_dict = self.dataset[idx]
#         # print(f"Processing video {idx}: {video_dict}")  # 디버깅 메시지 추가
          
#         # videoid, youtube_id, name = video_dict['audiocap_id'], video_dict['youtube_id'], video_dict['caption']
        
#         # video_filename = f"audiocaps_{youtube_id}.mp4"
#         # video_dir = os.path.join(self.video_folder, video_filename)
        
#         # if not os.path.exists(video_dir):
#         #     print(f"Video file {video_dir} not found.")  # 파일 존재 여부 확인
#         #     return None, None

#         # video_reader = VideoReader(video_dir, ctx=cpu(0))
#         # video_length = len(video_reader)
        
#         # if not self.is_image:
#         #     clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
#         #     start_idx = random.randint(0, video_length - clip_length)
#         #     batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
#         # else:
#         #     batch_index = [random.randint(0, video_length - 1)]

#         # pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
#         # pixel_values = pixel_values / 255.
#         # del video_reader

#         # if self.is_image:
#         #     pixel_values = pixel_values[0]
        
#         # return pixel_values, name

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         while True:
#             try:
#                 pixel_values, name = self.get_batch(idx)
#                 if pixel_values is None:
#                     idx = random.randint(0, self.length-1)
#                     continue
#                 break
#             except Exception as e:
#                 print(f"Error processing video {idx}: {e}")  # 예외 처리 메시지 추가
#                 idx = random.randint(0, self.length-1)

#         pixel_values = self.pixel_transforms(pixel_values)
#         pixel_values = rearrange(pixel_values, "f c h w -> c f h w")
#         #print("pixel_values.shape: ", pixel_values.shape)
#         sample = dict(pixel_values=pixel_values, text=name)
#         return sample

# if __name__ == "__main__":
#     print("succcess")
#     dataset = WebVid10M(
#         csv_path="scripts/evaluation/data/audiocaps.csv",
#         video_folder="scripts/evaluation/data/test_trimmed_audiocaps",
#         sample_size=256,
#         sample_stride=4, sample_n_frames=16,
#         is_image=True,
#     )
    
#     import pdb
#     pdb.set_trace()
    
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16)
#     for idx, batch in enumerate(dataloader):
#         print(batch["pixel_values"].shape, len(batch["text"]))
        