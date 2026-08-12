import os

# Limit CPU thread memory BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import torch
import torch.nn as nn


# ============================================================
# VGG ENCODER
# ============================================================

class VGGEncoder(nn.Module):

    def __init__(self, vgg_path):

        super(VGGEncoder, self).__init__()

        self.vgg = nn.Sequential(

            nn.Conv2d(3, 3, (1, 1)),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                3,
                64,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                64,
                64,
                (3, 3)
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                64,
                128,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                128,
                128,
                (3, 3)
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                128,
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
                256,
                (3, 3)
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                256,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                (2, 2),
                (2, 2),
                (0, 0),
                ceil_mode=True
            ),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU(),

            nn.ReflectionPad2d(
                (1, 1, 1, 1)
            ),

            nn.Conv2d(
                512,
                512,
                (3, 3)
            ),

            nn.ReLU()
        )

        # ====================================================
        # LOAD VGG WEIGHTS
        # ====================================================

        print("Loading VGG weights...")

        vgg_state = torch.load(
            vgg_path,
            map_location="cpu"
        )

        self.vgg.load_state_dict(
            vgg_state
        )

        # Release temporary checkpoint
        del vgg_state

        gc.collect()

        # ====================================================
        # ONLY USE VGG UP TO RELU4_1
        # ====================================================

        enc_layers = list(
            self.vgg.children()
        )[:31]

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

        # The original VGG container is no longer needed.
        del self.vgg

        gc.collect()

        # ====================================================
        # FREEZE VGG
        # ====================================================

        for name in [
            'enc_1',
            'enc_2',
            'enc_3',
            'enc_4'
        ]:

            for param in getattr(
                self,
                name
            ).parameters():

                param.requires_grad = False


    # ========================================================
    # FORWARD
    # ========================================================

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

        return (
            h1,
            h2,
            h3,
            h4
        )


# ============================================================
# DECODER
# ============================================================

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
                mode='nearest'
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
                mode='nearest'
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
                mode='nearest'
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


    def forward(
        self,
        input
    ):

        return self.net(input)