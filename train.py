import argparse
import gc
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from torch.nn.functional import mse_loss

from utils.utils import *
from utils.models import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--content_dir',
        type=str,
        default='/Users/riyanshu07/Desktop/NST_CODE/content_data',
        help='Location of content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default='/Users/riyanshu07/Desktop/NST_CODE/style_data',
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default='/Users/riyanshu07/Desktop/NST_CODE/vgg_normalised.pth',
        help='Location of pre-trained VGG'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment1',
        help='Name of Experiment'
    )

    # Reduced from 512 -> 256 for low-memory deployment
    parser.add_argument(
        '--Final_size',
        type=int,
        default=256,
        help='Size of final image'
    )

    parser.add_argument(
        '--content_size',
        type=int,
        default=256,
        help='Size of content image'
    )

    parser.add_argument(
        '--style_size',
        type=int,
        default=256,
        help='Size of style image'
    )

    parser.add_argument(
        '--crop',
        default=True,
        action='store_true',
        help='Crop Image'
    )

    # Reduced from 4 -> 1
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch Size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning Rate'
    )

    parser.add_argument(
        '--lr_decay',
        type=float,
        default=5e-5,
        help='Learning Decay'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of Epochs'
    )

    parser.add_argument(
        '--content_weight',
        type=float,
        default=1.0,
        help='Content Weight'
    )

    parser.add_argument(
        '--style_weight',
        type=float,
        default=10,
        help='Style Weight'
    )

    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Log Interval'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=2,
        help='Save Interval'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume Training'
    )

    parser.add_argument(
        '--decoder_path',
        type=str,
        default='',
        help='Path to decoder checkpoint'
    )

    parser.add_argument(
        '--optimizer_path',
        type=str,
        default='',
        help='Path to optimizer checkpoint'
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    # ---------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # SAVE DIRECTORY
    # ---------------------------------------------------------

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments
    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key} : {value}\n")

    # ---------------------------------------------------------
    # TRANSFORMS
    # ---------------------------------------------------------

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.Final_size
    )

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.Final_size
    )

    # ---------------------------------------------------------
    # DATASETS
    # ---------------------------------------------------------

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    # ---------------------------------------------------------
    # DATALOADERS
    # ---------------------------------------------------------
    # num_workers=0 saves RAM on low-memory servers.
    # pin_memory=False avoids unnecessary memory usage.
    # ---------------------------------------------------------

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

    print(
        "Number of batches in content dataset:",
        len(content_loader)
    )

    print(
        "Number of batches in style dataset:",
        len(style_loader)
    )

    # ---------------------------------------------------------
    # MODELS
    # ---------------------------------------------------------

    print("Loading VGG encoder...")

    encoder = VGGEncoder(args.vgg).to(device)

    print("Loading decoder...")

    decoder = Decoder().to(device)

    # ---------------------------------------------------------
    # FREEZE VGG
    # ---------------------------------------------------------

    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch:
        1.0 / (1.0 + args.lr_decay * epoch)
    )

    # ---------------------------------------------------------
    # RESUME TRAINING
    # ---------------------------------------------------------

    if args.resume:

        if args.decoder_path:
            print("Loading decoder checkpoint...")

            decoder.load_state_dict(
                torch.load(
                    args.decoder_path,
                    map_location=device
                )
            )

        if args.optimizer_path:
            print("Loading optimizer checkpoint...")

            optimizer.load_state_dict(
                torch.load(
                    args.optimizer_path,
                    map_location=device
                )
            )

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    print("Training...")

    for epoch in range(args.epochs):

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        progress_bar = tqdm(
            zip(content_loader, style_loader),
            total=min(
                len(content_loader),
                len(style_loader)
            )
        )

        for content_batch, style_batch in progress_bar:

            # -------------------------------------------------
            # MOVE DATA TO DEVICE
            # -------------------------------------------------

            content_batch = content_batch.to(
                device,
                non_blocking=False
            )

            style_batch = style_batch.to(
                device,
                non_blocking=False
            )

            # -------------------------------------------------
            # EXTRACT CONTENT + STYLE FEATURES
            # -------------------------------------------------
            # VGG is frozen, so gradients are NOT required here.
            # This saves a significant amount of RAM.
            # -------------------------------------------------

            with torch.no_grad():

                c_feats = encoder(content_batch)

                s_feats = encoder(style_batch)

                # -------------------------------------------------
                # AdaIN
                # -------------------------------------------------

                t = adaptive_instance_normalization(
                    c_feats[-1],
                    s_feats[-1]
                )

            # -------------------------------------------------
            # DECODER
            # -------------------------------------------------

            g = decoder(t)

            # -------------------------------------------------
            # FEATURES OF GENERATED IMAGE
            # -------------------------------------------------
            # DO NOT use torch.no_grad() here.
            #
            # We need gradients to travel:
            #
            # loss -> VGG -> generated image -> decoder
            #
            # VGG parameters are frozen, but gradients with
            # respect to its input are still calculated.
            # -------------------------------------------------

            g_feats = encoder(g)

            # -------------------------------------------------
            # CONTENT LOSS
            # -------------------------------------------------

            loss_c = (
                mse_loss(
                    g_feats[-1],
                    t
                )
                * args.content_weight
            )

            # -------------------------------------------------
            # STYLE LOSS
            # -------------------------------------------------

            loss_s = torch.tensor(
                0.0,
                device=device
            )

            for g_f, s_f in zip(g_feats, s_feats):

                # Style features do not require gradients.
                # s_f already came from no_grad().
                g_mean, g_std = cal_mean_std(g_f)

                s_mean, s_std = cal_mean_std(s_f)

                loss_s = loss_s + mse_loss(
                    g_mean,
                    s_mean
                )

                loss_s = loss_s + mse_loss(
                    g_std,
                    s_std
                )

            loss_s = loss_s * args.style_weight

            # -------------------------------------------------
            # TOTAL LOSS
            # -------------------------------------------------

            loss = loss_c + loss_s

            # -------------------------------------------------
            # BACKPROPAGATION
            # -------------------------------------------------

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            optimizer.step()

            # -------------------------------------------------
            # LOGGING
            # -------------------------------------------------

            progress_bar.set_description(
                f"Loss: {loss.item():.4f} | "
                f"Content: {loss_c.item():.4f} | "
                f"Style: {loss_s.item():.4f}"
            )

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

            # -------------------------------------------------
            # FREE TEMPORARY REFERENCES
            # -------------------------------------------------

            del c_feats
            del s_feats
            del g_feats
            del t
            del g
            del loss
            del loss_c
            del loss_s

            del content_batch
            del style_batch

            # Only relevant when using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Python garbage collection
            gc.collect()

        # -----------------------------------------------------
        # LR SCHEDULER
        # -----------------------------------------------------

        scheduler.step()

        # -----------------------------------------------------
        # AVERAGE LOSSES
        # -----------------------------------------------------

        num_batches = len(
            content_loader
        )

        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        # -----------------------------------------------------
        # LOGGING
        # -----------------------------------------------------

        if (epoch + 1) % args.log_interval == 0:

            tqdm.write(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Loss: {running_loss:.4f} | "
                f"Content: {running_closs:.4f} | "
                f"Style: {running_sloss:.4f}"
            )

        # -----------------------------------------------------
        # SAVE CHECKPOINT
        # -----------------------------------------------------

        if (epoch + 1) % args.save_interval == 0:

            torch.save(
                decoder.state_dict(),
                save_dir /
                f"decoder_epoch_{epoch + 1}.pth"
            )

            torch.save(
                optimizer.state_dict(),
                save_dir /
                f"optimizer_epoch_{epoch + 1}.pth"
            )

            print(
                f"Checkpoint saved for epoch {epoch + 1}"
            )

            # -------------------------------------------------
            # SAVE SAMPLE IMAGE
            # -------------------------------------------------

            with torch.no_grad():

                output = torch.cat(
                    [
                        content_batch,
                        style_batch,
                        g
                    ],
                    dim=0
                )

                save_image(
                    output,
                    save_dir /
                    f"output_epoch_{epoch + 1}.png",
                    nrow=args.batch_size
                )

    print("Training Completed!")


if __name__ == "__main__":
    main()