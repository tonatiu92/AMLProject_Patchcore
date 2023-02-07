import torch
import numpy as np
import torch.nn.functional as F
import PIL
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import os
import tqdm

def bilinear_interpolate(inputs: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Bilinear interpolation is a method of interpolation that uses the values of a grid of surrounding points to estimate the value at an intermediate point. 
    It is a type of linear interpolation that uses a weighted average of the values of the four points that surround a target point to estimate the value at that target point.

    To perform bilinear interpolation, we first determine the values of the four points that surround the target point. 
    Let's call these points P1, P2, P3, and P4, with P1 being the point at the top left of the target point and P4 being the point at the bottom right. 
    The value at the target point can then be estimated as follows:
    
    
    To match the original
    input resolution, (we may want to use intermediate network
    features), we upscale the result by bi-linear interpolation.
    """
    # Get the input tensor dimensions
    batch_size, num_channels, height, width = inputs.size()

    # Calculate the height and width scaling factors
    height_scale = (height - 1) / (output_size[0] - 1)
    width_scale = (width - 1) / (output_size[1] - 1)

    # Create the output tensor and fill it with zeros
    output = torch.zeros(batch_size, num_channels, *output_size)

    # Iterate over the output tensor pixels and perform bilinear interpolation
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            # Calculate the coordinates of the input tensor pixels to use for
            # interpolation
            x1 = int(i * height_scale)
            x2 = min(x1 + 1, height -1)
            y1 = int(j * width_scale)
            y2 = min(y1 + 1, width -1)
   

            # Calculate the interpolation weights
            w1 = (x2 - i * height_scale) * (y2 - j * width_scale)
            w2 = (x2 - i * height_scale) * (j * width_scale - y1)
            w3 = (i * height_scale - x1) * (y2 - j * width_scale)
            w4 = (i * height_scale - x1) * (j * width_scale - y1)

            # Perform the interpolation
            output[:, :, i, j] = w1 * inputs[:, :, x1, y1] + w2 * inputs[:, :, x1, y2] + w3 * inputs[:, :, x2, y1] + w4 * inputs[:, :, x2, y2]

    return output

def image_transform(datasets, data,image):
    datasets[data]["test"].dataloader.dataset.transform_std = [0.229, 0.224, 0.225]
    datasets[data]["test"].dataloader.dataset.transform_mean = [0.485, 0.456, 0.406]
    in_std = np.array(datasets[data]["test"].dataloader.dataset.transform_std).reshape(-1, 1, 1)
    in_mean = np.array( datasets[data]["test"].dataloader.dataset.transform_mean).reshape(-1, 1, 1)
    image = datasets[data]["test"].dataloader.dataset.transform_img(image)
    return np.clip( (image.numpy() * in_std + in_mean) * 255, 0, 255).astype(np.uint8)

def mask_transform(datasets, data,mask):
    return datasets[data]["test"].dataloader.dataset.transform_mask(mask).numpy()
    
def ROC(label, preds, data, clip = False):
    RocCurveDisplay.from_predictions(
        label,
        preds,
        name=f"anomaly vs non anomaly",
        color="darkorange",
    )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Good vs Anomaly - {data}")
    plt.legend()
    if clip:
        plt.savefig(f'.\\data\\{data}\\roc_auc_plot_clip.png')
    else:
        plt.savefig(f'.\\data\\{data}\\roc_auc_plot.png')
    plt.show()
    
def plot_segmentation_images(
    datasets,
    data,
    savefolder,
    image_paths,
    segmentations,
    anomaly_scores=None,
    mask_paths=None,
    save_depth=4,
):
    """Generate anomaly segmentation images.

    Args:
        image_paths: List[str] List of paths to images.
        segmentations: [List[np.ndarray]] Generated anomaly segmentations.
        anomaly_scores: [List[float]] Anomaly scores for each image.
        mask_paths: [List[str]] List of paths to ground truth masks.
        save_depth: [int] Number of path-strings to use for image savenames.
    """
    if mask_paths is None:
        mask_paths = ["-1" for _ in range(len(image_paths))]
    masks_provided = mask_paths[0] != "-1"
    if anomaly_scores is None:
        anomaly_scores = ["-1" for _ in range(len(image_paths))]

    os.makedirs(savefolder, exist_ok=True)
    i = 0
    for image_path, mask_path, anomaly_score, segmentation in tqdm.tqdm(
        zip(image_paths, mask_paths, anomaly_scores, segmentations),
        total=len(image_paths),
        desc="Generating Segmentation Images...",
        leave=False,
    ):
        
        image = PIL.Image.open(image_path).convert("RGB")
        image = image_transform(datasets,data,image)
        if not isinstance(image, np.ndarray):
            image = image.numpy()

        if masks_provided:
            if mask_path is not None:
                mask = PIL.Image.open(mask_path).convert("RGB")
                mask = mask_transform(datasets,data,mask)
                if not isinstance(mask, np.ndarray):
                    mask = mask.numpy()
            else:
                mask = np.zeros_like(image)
        savename = image_path.split("/")
        savename = "_".join(savename[-save_depth:])
        savename = os.path.join(savefolder, savename)
        f, axes = plt.subplots(1, 2 + int(masks_provided))
        axes[0].imshow(image.transpose(1, 2, 0))
        axes[1].imshow(mask.transpose(1, 2, 0))
        axes[2].imshow(segmentation)
        f.set_size_inches(3 * (2 + int(masks_provided)), 3)
        f.tight_layout()
        save = os.path.join(savefolder, str(i))
        f.savefig(save)
        i+=1
        plt.close()
