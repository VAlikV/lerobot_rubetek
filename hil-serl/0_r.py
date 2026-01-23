import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig

import torchvision.transforms as transforms

import time
import matplotlib.pyplot as plt

def main():
    # Device to use for training
    device = "mps"  # or "cuda", or "cpu"

    # Load the dataset used for training
    # repo_id = "lerobot/example_hil_serl_dataset"
    repo_id = "local/test_data"
    dataset = LeRobotDataset(repo_id)

    tr = transforms.ToPILImage()

    print(dataset[0].keys())

    plt.ion()  # включаем интерактивный режим
    fig, ax = plt.subplots()

    for i in range(len(dataset)):
        image = tr(dataset[i]['observation.image.side'])
        ax.clear()
        ax.imshow(image)
        ax.axis("off")
        plt.pause(0.2)
        print(i)
        print("Reward:", dataset[i]['next.reward'])
        print("Done:", dataset[i]['next.done'])
        # print("Penalty", dataset[i]['complementary_info.discrete_penalty'])
        print("Action", dataset[i]['action'])
        # print(dataset[i].keys())
        print()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
