import os
from tqdm import tqdm
import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
import argparse

from torchvision import transforms
from snntorch import surrogate
from torchvision import datasets
import snntorch.utils


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--time", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--beta", type=float, default=0.75)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--spike_prob", type=float, default=0.8)
parser.add_argument("--hidden_neurons", type=int, default=800)
parser.add_argument(
    '--subseq_spike_update',
    default=False,
    type=lambda x: (str(x).lower() in ['true', '1', 'yes']))

parser.set_defaults()

args = parser.parse_args()

time = args.time
batch_size = args.batch_size
beta = args.beta
lr = args.lr
n_epochs = args.n_epochs
spike_prob = args.spike_prob
hidden_neurons = args.hidden_neurons
subseq_spike_update = args.subseq_spike_update

spike_grad = surrogate.sigmoid(slope=5)

device = "cuda" if torch.cuda.is_available() else "cpu"

lif1 = snn.Leaky(beta=beta, init_hidden=True,
                 spike_grad=spike_grad,
                 ).to(torch.device(device))

lif2 = snn.Leaky(beta=beta, init_hidden=True,
                 spike_grad=spike_grad,
                 output=True,
                 ).to(torch.device(device))

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(28*28, hidden_neurons),
                    lif1,
                    nn.Linear(hidden_neurons, 10),
                    lif2).to(torch.device(device))

spike_recording = []

# correct class should fire 80% of the time
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.1)

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

# Load MNIST train data.
train_dataset = datasets.MNIST(
    root=os.path.join("data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0), (1)),
        ]
    ),
)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

# Load MNIST test data.
test_dataset = datasets.MNIST(
    root=os.path.join("data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0), (1)),
        ]
    ),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
    )

optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))


for epoch in range(n_epochs):

    loss_avg = torch.zeros((1), dtype=torch.float, device=device)
    print("Training...")
    acc = 0
    for step, batch in enumerate(tqdm(train_loader)):
        # initialize the total loss value
        loss_val = torch.zeros((1), dtype=torch.float, device=device)
        x, target = batch
        x = snn.spikegen.rate(spike_prob*x.squeeze(), num_steps=time)
        x = x.to(device)
        target = target.to(device)
        net.train()
        spike_record = []
        voltage = []
        v_seq = []
        snn.utils.reset(net)
        # apply the backward pass only after K time steps
        K = 10
        loss_seq = torch.zeros((1), dtype=torch.float, device=device)
        count = 1
        for t in range(time):
            s, v = net(x[t])
            spike_record.append(s)
            voltage.append(v)
            v_seq.append(s)

            if count == K and subseq_spike_update:

                loss = loss_fn(
                    torch.stack(v_seq),
                    target,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                for i, layer in enumerate(net):
                    if isinstance(layer, snn.Leaky):
                        net[i].detach_hidden()
                count = 1
                v_seq = []
            else:
                count += 1
        if not subseq_spike_update:
            loss = loss_fn(
                    torch.stack(v_seq),
                    target,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val += loss.item()

        acc += SF.accuracy_rate(torch.stack(spike_record), target)
        if step % 50 == 0 and step != 0:
            print(f"Train Subset Accuracy: {100 * acc / 50:.2f}%")
            acc = 0

        loss_avg += loss_val

    # Test loop
    print("Testing ...")
    total = 0
    acc = 0
    with torch.no_grad():
        net.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            data, targets = batch
            data = snn.spikegen.rate(data.squeeze(), num_steps=time)
            data = data.to(torch.device(device))
            targets = targets.to(torch.device(device))
            spike_record = []
            voltage = []
            # forward pass
            for t in range(time):
                s, v = net(data[t])
                spike_record.append(s)
                voltage.append(v)
            spike_record = torch.stack(spike_record)
            voltage = torch.stack(voltage)
            acc += SF.accuracy_rate(spike_record, targets)
            total += 1
        print(f"Test Set Accuracy: {100 * acc / total:.2f}%")

torch.save(net.state_dict(), "mnist_net_800n_norm.pt")
