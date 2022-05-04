import os
from tqdm import tqdm
import torch

import snntorch as snn
import snntorch.functional as SF
import argparse
import snntorch.utils

from torchvision import transforms

from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from networks import CSNN


parser = argparse.ArgumentParser()

parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--time", type=int, default=75)
parser.add_argument("--update_steps", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=0.75)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--thr", type=float, default=0.75)
parser.add_argument(
    '--enable_dropout',
    default=False,
    type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
parser.add_argument("--out", type=str, default="vgg9-f.pt")

parser.set_defaults()

args = parser.parse_args()

time = args.time
batch_size = args.batch_size
beta = args.beta
lr = args.lr
n_epochs = args.n_epochs
thr = args.thr
out = args.out
update_steps = args.update_steps
enable_dropout = args.enable_dropout

device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float


class BiPolar(torch.nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, img):
        if type(img) is not torch.Tensor:
            raise ValueError("Expected torch.Tensor, got ",
                             type(img), " instead")

        on = (img.sign() > 0).to(torch.float) * torch.abs(img)
        off = (img.sign() <= 0).to(torch.float) * torch.abs(img)
        return torch.cat((on, off), dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__
        return format_string


torch.manual_seed(17)

net = CSNN(time=time, beta=beta, thr=thr, n_chan=6)
net = net.to(torch.device(device))
spike_recording = []

loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.1)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

# Load CIFAR-10 train data
train_dataset = datasets.CIFAR10(
    root=os.path.join("data", "CIFAR10"),
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(size=[32, 32], scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
               mean=[0.5, 0.5, 0.5],
               std=[0.5, 0.5, 0.5]),
            BiPolar(),
        ]
    ),
)

# Load CIFAR-10 test data
test_dataset = datasets.CIFAR10(
    root=os.path.join("data", "CIFAR10"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
            BiPolar(),
        ]
    ),
)


train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
    )

writer = SummaryWriter()

for epoch in range(n_epochs):

    loss_avg = torch.zeros((1), dtype=torch.float, device=device)
    print("Training...")
    acc = 0
    for step, batch in enumerate(tqdm(train_loader)):
        # initialize the total loss value
        loss_val = torch.zeros((1), dtype=torch.float, device=device)
        x, target = batch
        x = snn.spikegen.rate(x, num_steps=time)
        x = x.to(torch.device(device))
        target = target.to(torch.device(device))
        net.train()

        snn.utils.reset(net)

        spike_record, voltage = net(x)

        loss = loss_fn(
                voltage,
                target,
            )

        writer.add_scalar(
                'Loss', loss, epoch*len(train_loader) + step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += SF.accuracy_rate(spike_record, target)

        if step % update_steps == 0 and step > 0:

            writer.add_scalar(
                'Accuracy (train)', 100 * acc / update_steps,
                epoch*len(train_loader) + step)
            acc = 0
            torch.save(net.state_dict(), out)

        loss_avg += loss.item()

    # Test loop
    print("Testing ...")
    total = 0
    acc = 0
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            data, targets = batch
            data = snn.spikegen.rate(data, num_steps=time)
            data = data.to(torch.device(device))
            targets = targets.to(torch.device(device))

            spike_record, voltage = net(data)
            acc += SF.accuracy_rate(spike_record, targets)
            total += 1
        print(f"Test Set Accuracy: {100 * acc / total:.2f}%")
        writer.add_scalar(
                'Accuracy (test)', 100 * acc / total, epoch)

print("\n Done!")
