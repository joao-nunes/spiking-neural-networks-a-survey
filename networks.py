import torch
import torch.nn as nn
import snntorch as snn

from snntorch import surrogate


class VGG9f(nn.Module):

    def __init__(self,
                 time: int = 75,
                 thr: float = 1.0,
                 beta: float = 0.25,
                 p_drop: float = 0.25,
                 enable_dropout: bool = False,
                 n_chan=1,
                 ):
        super(VGG9f, self).__init__()

        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.time = time

        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif2 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif3 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif4 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif5 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif6 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )
        self.lif7 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              )

        self.lif1_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif2_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif3_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif4_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif5_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )

        self.conv1 = nn.Conv2d(n_chan, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.p_drop = p_drop
        self.enable_dropout = enable_dropout
        self.fc1 = nn.Linear(2048, 10)

    def __dropout(self,
                  drop,
                  input,
                  training: bool = True,
                  enable: bool = True,):
        if training and enable:
            return drop(input)
        else:
            return input

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()

        mem1_1 = self.lif1_1.init_leaky()
        mem2_1 = self.lif2_1.init_leaky()
        mem3_1 = self.lif3_1.init_leaky()
        mem4_1 = self.lif4_1.init_leaky()
        mem5_1 = self.lif5_1.init_leaky()

        # Record the final layer
        spike_record = []
        voltage = []
        dropout_layers = [nn.Dropout(self.p_drop) for _ in range(6)]
        batch_size = x.shape[1]
        for t in range(self.time):

            x1 = self.conv1_1(self.conv1(x[t]))
            spk1, mem1 = self.lif1(x1, mem1)
            x2 = self.pool1(spk1)
            spk1_1, mem1_1 = self.lif1_1(x2, mem1_1)

            x3 = self.conv2_1(self.conv2(self.__dropout(
                dropout_layers[0],
                spk1_1,
                self.training,
                enable=self.enable_dropout,
                )))

            spk2, mem2 = self.lif2(x3, mem2)
            x4 = self.pool2(spk2)
            spk2_1, mem2_1 = self.lif2_1(x4, mem2_1)

            x5 = self.conv3(self.__dropout(
                dropout_layers[1],
                spk2_1,
                self.training
                ))
            spk3, mem3 = self.lif3(x5, mem3)
            x6 = self.pool3(spk3)
            spk3_1, mem3_1 = self.lif3_1(x6, mem3_1)

            x7 = self.conv4(self.__dropout(
                dropout_layers[2],
                spk3_1,
                self.training,
                enable=self.enable_dropout,
                ))
            spk4, mem4 = self.lif4(x7, mem4)
            x8 = self.pool4(spk4)
            spk4_1, mem4_1 = self.lif4_1(x8, mem4_1)

            x9 = self.conv5(self.__dropout(
                dropout_layers[3],
                spk4_1,
                self.training,
                enable=self.enable_dropout,
                ))
            spk5, mem5 = self.lif5(x9, mem5)
            x10 = self.pool5(spk5)
            spk5_1, mem5_1 = self.lif5_1(x10, mem5_1)

            x11 = self.conv6(self.__dropout(
                dropout_layers[4],
                spk5_1,
                self.training,
                enable=self.enable_dropout,
                ))
            spk6, mem6 = self.lif6(x11, mem6)

            x12 = self.fc1(self.__dropout(
                dropout_layers[5],
                spk6,
                self.training,
                enable=self.enable_dropout,
                ).view(batch_size, -1))

            spk7, mem7 = self.lif7(x12, mem7)

            spike_record.append(spk7)
            voltage.append(mem7)

        return torch.stack(spike_record), torch.stack(voltage)


class VGG9c(nn.Module):

    def __init__(self,
                 time: int = 75,
                 thr: float = 1.0,
                 beta: float = 0.25,
                 p_drop: float = 0.25,
                 enable_dropout: bool = False,
                 n_chan=1,
                 ):
        super(VGG9c, self).__init__()

        spike_grad = surrogate.fast_sigmoid(slope=5)
        self.time = time

        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif2 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif3 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif4 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif5 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif6 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )
        self.lif7 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              learn_threshold=True,
                              learn_beta=True,
                              )

        self.lif1_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif2_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif3_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif4_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )
        self.lif5_1 = snn.Leaky(beta=beta,
                                spike_grad=spike_grad,
                                threshold=thr-0.25,
                                )

        self.conv1 = nn.Conv2d(n_chan, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)

        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.p_drop = p_drop
        self.enable_dropout = enable_dropout
        self.fc1 = nn.Linear(1024*2*2, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def __dropout(self,
                  drop,
                  input,
                  training: bool = True,
                  enable: bool = True,):
        if training and enable:
            return drop(input)
        else:
            return input

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()
        mem7 = self.lif7.init_leaky()

        mem1_1 = self.lif1_1.init_leaky()
        mem2_1 = self.lif2_1.init_leaky()
        mem3_1 = self.lif3_1.init_leaky()
        mem4_1 = self.lif4_1.init_leaky()

        # Record the final layer
        spike_record = []
        voltage = []
        dropout_layers = [nn.Dropout(self.p_drop) for _ in range(6)]
        batch_size = x.shape[1]
        for t in range(self.time):

            x1 = self.conv1_1(self.conv1(x[t]))
            spk1, mem1 = self.lif1(x1, mem1)
            x2 = self.pool1(spk1)
            spk1_1, mem1_1 = self.lif1_1(x2, mem1_1)

            x3 = self.conv2_1(self.conv2(self.__dropout(
                dropout_layers[0],
                spk1_1,
                self.training,
                enable=self.enable_dropout,
                )))

            spk2, mem2 = self.lif2(x3, mem2)
            x4 = self.pool2(spk2)
            spk2_1, mem2_1 = self.lif2_1(x4, mem2_1)

            x5 = self.conv3(self.__dropout(
                dropout_layers[1],
                spk2_1,
                self.training
                ))
            spk3, mem3 = self.lif3(x5, mem3)
            x6 = self.pool3(spk3)
            spk3_1, mem3_1 = self.lif3_1(x6, mem3_1)

            x7 = self.conv4(self.__dropout(
                dropout_layers[2],
                spk3_1,
                self.training,
                enable=self.enable_dropout,
                ))
            spk4, mem4 = self.lif4(x7, mem4)
            x8 = self.pool4(spk4)
            spk4_1, mem4_1 = self.lif4_1(x8, mem4_1)

            x9 = self.conv5(self.__dropout(
                dropout_layers[3],
                spk4_1,
                self.training,
                enable=self.enable_dropout,
                ))
            spk5, mem5 = self.lif5(x9, mem5)

            x10 = self.fc1(self.__dropout(
                dropout_layers[5],
                spk5,
                self.training,
                enable=self.enable_dropout,
                ).view(batch_size, -1))

            spk6, mem6 = self.lif7(x10, mem6)
            x11 = self.fc2(spk6)
            spk7, mem7 = self.lif7(x11, mem7)
            spike_record.append(spk7)
            voltage.append(mem7)

        return torch.stack(spike_record), torch.stack(voltage)


class CSNN(nn.Module):

    def __init__(
        self,
        time=25,
        beta=0.5,
        n_chan=6,
        thr=1.0,
        device=torch.device('cpu')
    ):

        super(CSNN, self).__init__()

        spike_grad = surrogate.fast_sigmoid(slope=5)

        self.conv1 = nn.Conv2d(n_chan, 24, 5)
        self.pool = nn.MaxPool2d((2, 2))

        self.lif1 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              ).to(torch.device(device))

        self.conv2 = nn.Conv2d(24, 128, 5)

        self.lif2 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              ).to(torch.device(device))

        self.fc1 = nn.Linear(128 * 5 * 5, 10)

        self.lif3 = snn.Leaky(beta=beta,
                              spike_grad=spike_grad,
                              threshold=thr,
                              ).to(torch.device(device))
        self.time = time

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        # Record the final layer
        spike_record = []
        voltage = []
        batch_size = x.shape[1]
        for t in range(self.time):
            cur1 = self.pool(self.conv1(x[t, ...]))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spike_record.append(spk3)
            voltage.append(mem3)

        return torch.stack(spike_record), torch.stack(voltage)
