import os
import gc

import torch
from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# ============================================================
# APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get(
    'SECRET_KEY',
    'supersecretkey'
)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

app.config['ALLOWED_EXTENSIONS'] = {
    'png',
    'jpg',
    'jpeg'
}

Bootstrap(app)

os.makedirs(
    app.config['UPLOAD_FOLDER'],
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

# Render will normally use CPU.
# Keep CPU thread count low to reduce memory usage.

torch.set_num_threads(1)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, 'mps')
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ============================================================
# FORM
# ============================================================

class UploadForm(FlaskForm):

    content = FileField('Content Image')

    style = FileField('Style Image')

    content_path = HiddenField()

    style_path = HiddenField()

    alpha = FloatField(
        'Alpha',
        default=1.0
    )

    submit = SubmitField(
        'Transfer Style'
    )


# ============================================================
# MODEL PATHS
# ============================================================

VGG_PATH = "vgg_normalised.pth"

PRETRAINED_DECODER = "pretrained_decoder.pth"

FALLBACK_DECODER = (
    "experiment/experiment3/decoder_epoch_10.pth"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading VGG encoder...")

encoder = VGGEncoder(
    VGG_PATH
).to(device)

encoder.eval()

# Freeze VGG completely.
# We only use it as a feature extractor.

for param in encoder.parameters():
    param.requires_grad = False


print("VGG encoder loaded.")


print("Loading decoder...")

decoder = Decoder().to(device)


# ------------------------------------------------------------
# Select decoder checkpoint
# ------------------------------------------------------------

if os.path.exists(PRETRAINED_DECODER):

    decoder_path = PRETRAINED_DECODER

elif os.path.exists(FALLBACK_DECODER):

    decoder_path = FALLBACK_DECODER

else:

    raise FileNotFoundError(
        "No decoder checkpoint found. "
        "Expected pretrained_decoder.pth or "
        "experiment/experiment3/decoder_epoch_10.pth"
    )


print(
    f"Loading decoder checkpoint: {decoder_path}"
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

raw_state = torch.load(
    decoder_path,
    map_location=device
)


# Your decoder checkpoint sometimes contains
# keys such as:
#
# conv1.weight
#
# while the Decoder expects:
#
# net.conv1.weight
#
# Therefore normalize the keys.

state = {}

for key, value in raw_state.items():

    if key.startswith("net."):

        state[key] = value

    else:

        state["net." + key] = value


decoder.load_state_dict(
    state,
    strict=True
)


# ------------------------------------------------------------
# VERY IMPORTANT:
# Release checkpoint memory after loading.
# ------------------------------------------------------------

del raw_state
del state

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


decoder.eval()


for param in decoder.parameters():
    param.requires_grad = False


print("Decoder loaded successfully.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (256, 256)
    ),
    transforms.ToTensor()
])


# ============================================================
# FILE VALIDATION
# ============================================================

def allowed_file(filename):

    return (
        '.' in filename
        and
        filename.rsplit(
            '.',
            1
        )[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
    )


# ============================================================
# STYLE TRANSFER
# ============================================================

def style_transfer(
    content_image,
    style_image,
    alpha
):

    # --------------------------------------------------------
    # Convert images to tensors
    # --------------------------------------------------------

    content_tensor = transform(
        content_image
    ).unsqueeze(0)

    style_tensor = transform(
        style_image
    ).unsqueeze(0)


    # Move to device

    content_tensor = content_tensor.to(device)

    style_tensor = style_tensor.to(device)


    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------
    #
    # inference_mode() uses less memory than normal autograd
    # because this application does NOT train the model.
    # --------------------------------------------------------

    with torch.inference_mode():

        # VGG features

        content_feats = encoder(
            content_tensor,
            is_test=True
        )

        style_feats = encoder(
            style_tensor,
            is_test=True
        )


        # ----------------------------------------------------
        # AdaIN
        # ----------------------------------------------------

        stylized_feats = (
            adaptive_instance_normalization(
                content_feats,
                style_feats
            )
        )


        # ----------------------------------------------------
        # Alpha blending
        # ----------------------------------------------------

        stylized_feats = (
            alpha * stylized_feats
            +
            (1.0 - alpha) * content_feats
        )


        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        stylized_image = decoder(
            stylized_feats
        )


    # --------------------------------------------------------
    # Release input tensors
    # --------------------------------------------------------

    del content_tensor
    del style_tensor
    del content_feats
    del style_feats
    del stylized_feats

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    image.save(
        path
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    '/',
    methods=['GET', 'POST']
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

            if allowed_file(
                form.content.data.filename
            ):

                content_filename = secure_filename(
                    form.content.data.filename
                )

                content_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
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

            if allowed_file(
                form.style.data.filename
            ):

                style_filename = secure_filename(
                    form.style.data.filename
                )

                style_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
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


        # ====================================================
        # RUN STYLE TRANSFER
        # ====================================================

        if (
            content_filename
            and
            style_filename
        ):

            content_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                content_filename
            )

            style_path = os.path.join(
                app.config['UPLOAD_FOLDER'],
                style_filename
            )

            try:

                # --------------------------------------------
                # Open images
                # --------------------------------------------

                with Image.open(
                    content_path
                ) as content_image:

                    content_image = (
                        content_image
                        .convert('RGB')
                    )


                    with Image.open(
                        style_path
                    ) as style_image:

                        style_image = (
                            style_image
                            .convert('RGB')
                        )


                        # ------------------------------------
                        # Alpha
                        # ------------------------------------

                        alpha = (
                            float(form.alpha.data)
                            if form.alpha.data
                            is not None
                            else 1.0
                        )


                        # Keep alpha in valid range

                        alpha = max(
                            0.0,
                            min(
                                1.0,
                                alpha
                            )
                        )


                        # ------------------------------------
                        # Style transfer
                        # ------------------------------------

                        stylized_image = style_transfer(
                            content_image,
                            style_image,
                            alpha
                        )


                # --------------------------------------------
                # Output path
                # --------------------------------------------

                result_filename = (
                    'stylized_'
                    + content_filename
                )

                result_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    result_filename
                )


                # --------------------------------------------
                # Save output
                # --------------------------------------------

                save_image(
                    stylized_image,
                    result_path
                )


                result_image = (
                    result_filename
                )


                # --------------------------------------------
                # Release generated tensor
                # --------------------------------------------

                del stylized_image

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            except Exception as e:

                print(
                    f"Style transfer error: {e}"
                )

                error = str(e)


        else:

            if not content_filename:

                error = (
                    'Please upload content image'
                )

            elif not style_filename:

                error = (
                    'Please upload style image'
                )


    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


# ============================================================
# UPLOADS
# ============================================================

@app.route(
    '/uploads/<filename>'
)
def send_image(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )


# ============================================================
# EXAMPLES
# ============================================================

@app.route(
    '/examples/<path:filename>'
)
def send_example(filename):

    return send_from_directory(
        'examples',
        filename
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == '__main__':

    port = int(
        os.environ.get(
            'PORT',
            5001
        )
    )

    app.run(
        host='0.0.0.0',
        port=port
    )