import torch
import torch.nn as nn


class VGGEncoder(nn.Module):

    def __init__(self, vgg_path):
        super(VGGEncoder, self).__init__()

        # Only build layers up to relu4-1.
        # AdaIN does not use relu4-2, relu4-3, relu4-4
        # or any of the VGG block 5 layers.

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU()  # relu4-1
        )

        # --------------------------------------------------
        # Load only the required VGG weights
        # --------------------------------------------------

        state_dict = torch.load(
            vgg_path,
            map_location="cpu",
            weights_only=True
        )

        # The checkpoint contains the full VGG.
        # We only need layers 0 through 30.
        required_keys = {}

        for key, value in state_dict.items():

            # Example:
            # "0.weight"
            # "2.weight"
            # etc.

            parts = key.split(".")

            try:
                layer_index = int(parts[0])
            except ValueError:
                continue

            if layer_index < 31:
                required_keys[key] = value

        self.vgg.load_state_dict(
            required_keys,
            strict=True
        )

        # Release checkpoint memory immediately.
        del state_dict
        del required_keys

        # --------------------------------------------------
        # Split encoder into AdaIN blocks
        # --------------------------------------------------

        enc_layers = list(
            self.vgg.children()
        )

        self.enc_1 = nn.Sequential(
            *enc_layers[:4]
        )

        self.enc_2 = nn.Sequential(
            *enc_layers[4:11]
        )

        self.enc_3 = nn.Sequential(
            *enc_layers[11:18]
        )

        self.enc_4 = nn.Sequential(
            *enc_layers[18:31]
        )

        # We no longer need the combined VGG container.
        del self.vgg

        # --------------------------------------------------
        # Freeze encoder
        # --------------------------------------------------

        for name in [
            "enc_1",
            "enc_2",
            "enc_3",
            "enc_4"
        ]:

            for param in getattr(
                self,
                name
            ).parameters():

                param.requires_grad = False

    def forward(
        self,
        input,
        is_test=False
    ):

        h1 = self.enc_1(input)

        h2 = self.enc_2(h1)

        h3 = self.enc_3(h2)

        h4 = self.enc_4(h3)

        if is_test:
            return h4

        return h1, h2, h3, h4


class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()

        self.net = nn.Sequential(

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                256,
                (3, 3)
            ),

            nn.ReLU(),

            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                256,
                256,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                256,
                256,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                256,
                256,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                256,
                128,
                (3, 3)
            ),

            nn.ReLU(),

            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                128,
                128,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                128,
                64,
                (3, 3)
            ),

            nn.ReLU(),

            nn.Upsample(
                scale_factor=2,
                mode="nearest"
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                64,
                64,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                64,
                3,
                (3, 3)
            )
        )

    def forward(self, input):

        return self.net(input)