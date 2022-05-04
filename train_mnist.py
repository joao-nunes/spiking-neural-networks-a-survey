from os.path import dirname as up
from torchvision import transforms
from tqdm import tqdm
from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder, poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015

from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

import pickle as cPickle
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=800)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--p_drop", type=float, default=0.2)
parser.add_argument("--model_name", type=str,
                    default="mnist_diehl_and_cook_2015")
parser.set_defaults(plot=False, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
p_drop = args.p_drop
model_name = args.model_name

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=28*28,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    theta_plus=theta_plus,
    wmin=0,
    wmax=1,
    norm=78.4,
    inpt_shape=(1, 28, 28),
)


# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * intensity)]
    ),
)


# Record spikes during the simulation.
spike_record = torch.zeros(
    (update_interval, int(time / dt), n_neurons),
    device=device)

# Neuron assignments and spike proportions.
n_classes = 10

assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)

rates = torch.zeros((n_neurons, n_classes), device=device)


# Sequence of accuracy estimates.

accuracy = {"all": [], "proportion": []}
best_accuracy = -1
# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer],
        state_vars=["s"],
        time=int(time / dt),
        device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer],
        state_vars=["v"],
        time=int(time / dt),
        device=device
    )

    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin training.\n")
start = t()
labels = []
for epoch in range(n_epochs):

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch,
                                                    n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.

        inputs = {"X": batch["encoded_image"].view(
            int(time / dt), 1, 1, 28, 28
        )}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        if step % update_interval == 0:
            network.save(
                os.path.join(
                    up(up(up(__file__))),
                    "results",
                    "parameters",
                    model_name))

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=assignments,
                n_labels=n_classes,
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

# Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f"
                "(last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f"
                "(last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

            if accuracy["all"][-1] > best_accuracy:

                print("Accuracy improved! Saving best model parameters...")
                best_accuracy = accuracy["all"][-1]
                network.save(os.path.join(
                    up(up(up(__file__))),
                    "results",
                    "parameters",
                    model_name+"_best.pt",))

                with open(os.path.join(
                     up(up(up(__file__))),
                     "results",
                     "parameters",
                     "assignments_"+model_name+"_best.pickle"),
                     'wb') as fp:

                    cPickle.dump(assignments.detach().cpu().numpy(), fp, -1)

                with open(os.path.join(
                     up(up(up(__file__))),
                     "results",
                     "parameters",
                     "proportions_"+model_name+"_best.pickle"),
                     'wb') as fp:

                    cPickle.dump(proportions.detach().cpu().numpy(), fp, -1)

                print("Best model saved!")

            network.save(os.path.join(
                up(up(up(__file__))),
                "results",
                "parameters",
                model_name+".pt",))

            with open(os.path.join(
                 up(up(up(__file__))),
                 "results",
                 "parameters",
                 "assignments_"+model_name+".pickle"),
                 'wb') as fp:

                cPickle.dump(assignments.detach().cpu().numpy(), fp, -1)

            with open(os.path.join(
                 up(up(up(__file__))),
                 "results",
                 "parameters",
                 "proportions_"+model_name+".pickle"),
                 'wb') as fp:

                cPickle.dump(proportions.detach().cpu().numpy(), fp, -1)

            print('saved model!')

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        retries = 0
        while spikes['Ae'].get('s').sum() < 5 and retries < 3:
            network.reset_state_variables()
            retries += 1
            batch["image"] += 32
            sample = poisson(datum=batch["image"], time=time, dt=dt)
            inputs = {"X": sample.view(int(time / dt), 1, 1, 28, 28)}
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            network.run(inputs=inputs, time=time,  input_time_dim=1)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        spike_record[step % update_interval] = spikes[
                "Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(
                spikes_,
                ims=spike_ims,
                axes=spike_axes)

            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy,
                x_scale=update_interval,
                ax=perf_ax)

            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)
        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")
