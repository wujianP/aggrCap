import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self, caption_path, prompt_template=None):
        self.caption_path = caption_path
        self.prompt_template = prompt_template
        self.cap_dict = torch.load(caption_path)
        self.filenames = list(self.cap_dict.keys())

    def __len__(self):
        return len(self.cap_dict)

    def __getitem__(self, index):
        filename = self.filenames[index]
        caption = '"' + str(self.cap_dict[filename][::2]).replace('\'', '')[1:-1] + '"'
        action = filename.split('/')[-2].replace('_', ' ')
        # prompt = self.prompt_template + caption + f', {action}, \nOutput:'

        prompt = "Input: These are captions of the frames in temporal order within the same video:" + caption + ". please summarize the whole video according to the frame captions in short.Output: This video shows"

        # prompt = "Input: These are captions of the frames in temporal order within the same video:" + caption + "and the action of the video is " + action + ". please summarize the whole video according to the frame captions and actions in short.Output: This video shows"
        # prompt = "Input: These are captions of the frames in temporal order within the same video:" + caption + "and the action of the video is " + action + ". please generate the description of the whole video according to the frame captions and actions in short.Output: This video shows"
        # prompt = "Input: These are captions of the frames in temporal order within the same video:" + caption + "and the action of the video is " + action + ". please generate enrich the video description according to the frame captions and the video action in short.Output: This video shows"
        # prompt = "Input: These are captions of the frames in temporal order within the same video:" + caption + "and the action of the video is " + action + ". please generate the description of the whole video according to the action and the frame captions in short and in details.Output: "
        # prompt = self.prompt_template + caption + f', {action}, \nOutput: This video shows:'
        return filename, prompt, caption, action
