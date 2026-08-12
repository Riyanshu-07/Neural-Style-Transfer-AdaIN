import os
import torch

# IMPORTANT: set before loading any PyTorch models
torch.set_num_threads(1)

from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

# Your AdaIN code
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

VGG_PATH = os.path.join(
    BASE_DIR,
    "vgg_normalised.pth"
)

DECODER_PATH = os.path.join(
    BASE_DIR,
    "pretrained_decoder.pth"
)


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = "supersecretkey"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["ALLOWED_EXTENSIONS"] = {
    "png",
    "jpg",
    "jpeg"
}

Bootstrap(app)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

# Render Free has CPU only
device = torch.device("cpu")

print("========================================", flush=True)
print("Starting AdaIN Neural Style Transfer", flush=True)
print("Device:", device, flush=True)
print("PyTorch threads:", torch.get_num_threads(), flush=True)
print("========================================", flush=True)


# ============================================================
# FORM
# ============================================================

class UploadForm(FlaskForm):

    content = FileField("Content Image")

    style = FileField("Style Image")

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        "Alpha",
        default=1.0
    )

    submit = SubmitField(
        "Transfer Style"
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(VGG_PATH):
    raise FileNotFoundError(
        f"VGG model not found: {VGG_PATH}"
    )

if not os.path.exists(DECODER_PATH):
    raise FileNotFoundError(
        f"Decoder model not found: {DECODER_PATH}"
    )


# ============================================================
# LOAD VGG ENCODER
# ============================================================

print("Loading VGG encoder...", flush=True)

encoder = VGGEncoder(
    VGG_PATH
).to(device)

encoder.eval()

print(
    "VGG encoder loaded.",
    flush=True
)


# ============================================================
# LOAD DECODER
# ============================================================

print(
    "Creating decoder...",
    flush=True
)

decoder = Decoder().to(device)

print(
    "Loading decoder weights...",
    flush=True
)


state_dict = torch.load(
    DECODER_PATH,
    map_location=device,
    weights_only=True
)


# ============================================================
# FIX DECODER CHECKPOINT KEYS
# ============================================================

# Checkpoint contains:
#
# 1.weight
# 1.bias
# 5.weight
#
# Decoder expects:
#
# net.1.weight
# net.1.bias
# net.5.weight
#
# Therefore add "net." when required.

if isinstance(state_dict, dict):

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not any(
        key.startswith("net.")
        for key in state_dict.keys()
    ):

        state_dict = {
            "net." + key: value
            for key, value in state_dict.items()
        }


print(
    "Loading decoder state dict...",
    flush=True
)

decoder.load_state_dict(
    state_dict
)

decoder.eval()


# Release checkpoint memory
del state_dict


print(
    "Decoder loaded successfully.",
    flush=True
)

print("========================================", flush=True)
print("ALL MODELS LOADED SUCCESSFULLY", flush=True)
print("========================================", flush=True)


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )


# ============================================================
# ADAIN STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha
):

    # 256x256 keeps inference memory lower.
    content_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    # --------------------------------------------------------
    # Convert images to tensors
    # --------------------------------------------------------

    content_tensor = (
        content_transform(content_image)
        .unsqueeze(0)
        .to(device)
    )

    style_tensor = (
        style_transform(style_image)
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # AdaIN inference
    # --------------------------------------------------------

    with torch.no_grad():

        print(
            "Encoding content image...",
            flush=True
        )

        content_feats = encoder(
            content_tensor,
            is_test=True
        )


        print(
            "Encoding style image...",
            flush=True
        )

        style_feats = encoder(
            style_tensor,
            is_test=True
        )


        print(
            "Applying AdaIN...",
            flush=True
        )

        stylized_feats = (
            adaptive_instance_normalization(
                content_feats,
                style_feats
            )
        )


        # Alpha blending
        stylized_feats = (
            alpha * stylized_feats
            +
            (1.0 - alpha) * content_feats
        )


        print(
            "Decoding image...",
            flush=True
        )

        stylized_image = decoder(
            stylized_feats
        )


    # Release intermediate tensors
    del content_tensor
    del style_tensor
    del content_feats
    del style_feats
    del stylized_feats


    return stylized_image


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    image,
    path
):

    image = image.detach()

    image = image.cpu()

    image = image.squeeze(0)

    image = image.clamp(
        0,
        1
    )

    image = transforms.ToPILImage()(
        image
    )

    image.save(path)


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    form = UploadForm()

    result_image = None

    content_filename = None

    style_filename = None

    error = None


    # ========================================================
    # FORM SUBMISSION
    # ========================================================

    if form.validate_on_submit():

        # ----------------------------------------------------
        # CONTENT IMAGE
        # ----------------------------------------------------

        if (
            form.content.data
            and
            form.content.data.filename
        ):

            if not allowed_file(
                form.content.data.filename
            ):

                error = (
                    "Invalid content image format."
                )

            else:

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    UPLOAD_FOLDER,
                    content_filename
                )

                form.content.data.save(
                    content_path
                )

                form.content_path.data = (
                    content_filename
                )

        else:

            content_filename = (
                form.content_path.data
            )


        # ----------------------------------------------------
        # STYLE IMAGE
        # ----------------------------------------------------

        if (
            form.style.data
            and
            form.style.data.filename
        ):

            if not allowed_file(
                form.style.data.filename
            ):

                error = (
                    "Invalid style image format."
                )

            else:

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    UPLOAD_FOLDER,
                    style_filename
                )

                form.style.data.save(
                    style_path
                )

                form.style_path.data = (
                    style_filename
                )

        else:

            style_filename = (
                form.style_path.data
            )


        # ----------------------------------------------------
        # VALIDATE BOTH IMAGES
        # ----------------------------------------------------

        if not content_filename:

            error = (
                "Please upload a content image."
            )

        elif not style_filename:

            error = (
                "Please upload a style image."
            )


        # ----------------------------------------------------
        # RUN STYLE TRANSFER
        # ----------------------------------------------------

        if (
            content_filename
            and
            style_filename
            and
            error is None
        ):

            content_path = os.path.join(
                UPLOAD_FOLDER,
                content_filename
            )

            style_path = os.path.join(
                UPLOAD_FOLDER,
                style_filename
            )


            try:

                print(
                    "Opening images...",
                    flush=True
                )


                content_image = (
                    Image.open(
                        content_path
                    ).convert("RGB")
                )


                style_image = (
                    Image.open(
                        style_path
                    ).convert("RGB")
                )


                # ------------------------------------------------
                # Alpha
                # ------------------------------------------------

                alpha = (
                    form.alpha.data
                    if form.alpha.data is not None
                    else 1.0
                )

                alpha = float(alpha)

                alpha = max(
                    0.0,
                    min(
                        1.0,
                        alpha
                    )
                )


                # ------------------------------------------------
                # STYLE TRANSFER
                # ------------------------------------------------

                print(
                    "Starting AdaIN inference...",
                    flush=True
                )


                stylized_image = style_transfer(
                    content_image,
                    style_image,
                    alpha
                )


                # ------------------------------------------------
                # SAVE RESULT
                # ------------------------------------------------

                result_filename = (
                    "stylized_"
                    +
                    content_filename
                )

                result_path = os.path.join(
                    UPLOAD_FOLDER,
                    result_filename
                )


                save_image(
                    stylized_image,
                    result_path
                )


                result_image = (
                    result_filename
                )


                print(
                    "Style transfer completed!",
                    flush=True
                )


                # Release memory
                del content_image
                del style_image
                del stylized_image


            except Exception as e:

                print(
                    "STYLE TRANSFER ERROR:",
                    str(e),
                    flush=True
                )

                error = str(e)


    # ========================================================
    # RENDER TEMPLATE
    # ========================================================

    return render_template(
        "index.html",
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# SERVE UPLOADED IMAGES
# ============================================================

@app.route(
    "/static/uploads/<filename>"
)
def send_image(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# EXAMPLES
# ============================================================

@app.route(
    "/examples/<filename>"
)
def send_example(filename):

    examples_folder = os.path.join(
        BASE_DIR,
        "examples"
    )

    return send_from_directory(
        examples_folder,
        filename
    )


# ============================================================
# START FLASK
# ============================================================

if __name__ == "__main__":

    print(
        "Starting Flask server...",
        flush=True
    )

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False
    )