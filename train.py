import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from utils.utils import *
import torch.optim as optim
from utils.models import *
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from torch.nn.functional import mse_loss

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_dir',type=str,default='/Users/riyanshu07/Desktop/NST_CODE/content_data',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default='/Users/riyanshu07/Desktop/NST_CODE/style_data',help='Location of style dataset')
    parser.add_argument('--vgg',type=str,default='/Users/riyanshu07/Desktop/NST_CODE/vgg_normalised.pth',help='Location of pre-trained VGG')
    parser.add_argument('--experiment',type=str,default='experiment1',help='Name of Experiment')
    parser.add_argument('--Final_size',type=int,default=512,help='Size of Final Image')
    parser.add_argument('--content_size',type=int,default=512,help='Size of Final Image')
    parser.add_argument('--style_size',type=int,default=512,help='Size of Content Image')
    parser.add_argument('--crop',default=True,action='store_true',help='Crop Image')
    parser.add_argument('--batch_size',type=int,default=4,help='Batch Size')
    parser.add_argument('--lr',type=float,default=1e-4,help='Learning Rate')
    parser.add_argument('--lr_decay',type=float,default=5e-5,help='Learning Decay')
    parser.add_argument('--epochs',type=int,default=2,help='Number of Epoch')
    parser.add_argument('--content_weight',type=float,default=1.0,help='Content Weights')
    parser.add_argument('--style_weight',type=float,default=10,help='Style Weights')
    parser.add_argument('--log_interval',type=int,default=1,help='Log Interval')
    parser.add_argument('--save_interval',type=int,default=2,help='Save Interval')
    parser.add_argument('--resume',action='store_true',default =False,help='Resume Training')
    parser.add_argument('--decoder_path',type=str,default=2,help='path to decoder checkpoint')
    parser.add_argument('--optimizer_path',type=str,default=2,help='path to optimizer checkpoint')
    return parser.parse_args()
    
def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save arguments
    with open(save_dir / "args.txt", "w") as args_file:
        for key, value in vars(args).items():
            args_file.write(f"{key} : {value}\n")

    style_transform = get_transform(
        args.style_size, args.crop, args.Final_size
    )
    content_transform = get_transform(
        args.content_size, args.crop, args.Final_size
    )

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )   

    style_loader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    print("Number of batches in content dataset:", len(content_loader))
    print("Number of batches in style dataset:", len(style_loader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch),
    )

    if args.resume:
        decoder.load_state_dict(torch.load(args.decoder_path))
        optimizer.load_state_dict(torch.load(args.optimizer_path))

    print("Training...")

    encoder.eval()

    for epoch in range(args.epochs):

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        progress_bar = tqdm(
            zip(content_loader, style_loader),
            total=min(len(content_loader), len(style_loader)),
        )

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Extract features
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # AdaIN
            t = adaptive_instance_normalization(
                c_feats[-1], s_feats[-1]
            )

            # Generate stylized image
            g = decoder(t)

            # Extract generated image features
            g_feats = encoder(g)

            # Content Loss
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            # Style Loss
            loss_s = 0.0

            for g_f, s_f in zip(g_feats, s_feats):

                g_mean, g_std = cal_mean_std(g_f)
                s_mean, s_std = cal_mean_std(s_f)

                loss_s += mse_loss(g_mean, s_mean)
                loss_s += mse_loss(g_std, s_std)

            loss_s *= args.style_weight

            # Total Loss
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f"Loss: {loss.item():.4f} | "
                f"Content: {loss_c.item():.4f} | "
                f"Style: {loss_s.item():.4f}"
            )

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        running_loss /= len(content_loader)
        running_closs /= len(content_loader)
        running_sloss /= len(content_loader)

        if (epoch + 1) % args.log_interval == 0:

            tqdm.write(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Loss: {running_loss:.4f} | "
                f"Content: {running_closs:.4f} | "
                f"Style: {running_sloss:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0:

            torch.save(
                decoder.state_dict(),
                save_dir / f"decoder_epoch_{epoch+1}.pth",
            )

            torch.save(
                optimizer.state_dict(),
                save_dir / f"optimizer_epoch_{epoch+1}.pth",
            )

            with torch.no_grad():

                output = torch.cat(
                    [content_batch, style_batch, g],
                    dim=0,
                )

                save_image(
                    output,
                    save_dir / f"output_epoch_{epoch+1}.png",
                    nrow=args.batch_size,
                )

    print("Training Completed!")

if __name__ == '__main__':
    main()