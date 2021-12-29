import numpy as np
import base64
import os
import secrets
import argparse
from PIL import Image

import torch
from torch import nn
from training.model import Generator, Encoder
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import io
import cv2
# eye, nose, mouth
seg_index = {'e':[5,4],
            'n':[10],
            'm':[11,12,13]}

# for 1 gpu only.
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.g_ema = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul=train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )
        self.e_ema = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )
        self.device = device

    def forward(self, original_image, references, masks, shift_values):

        combined = torch.cat([original_image, references], dim=0) # shape = 2,256,256,3

        # original image & reference images 들의 stylemap을 한번에 뽑아낸다.
        ws = self.e_ema(combined)
        # original / reference stylemap을 분리한다.
        original_stylemap, reference_stylemaps = torch.split( 
            ws, [1, len(ws) - 1], dim=0
        )

        # mask를 정확히 알아 봐야 함
        mixed = self.g_ema(
            [original_stylemap, reference_stylemaps],
            input_is_stylecode=True,
            mix_space="demo",
            mask=[masks, shift_values, args.interpolation_step],
        )[0]

        return mixed

def mask2masks(mask,keys):
    mask = np.expand_dims(np.array(mask),axis=0)
    kernel = np.ones((3,3), np.uint8)
    context,label_mask = np.zeros_like(mask,dtype=np.float32), np.zeros_like(mask,dtype=np.float32)
    masks = [context,context,context]

    for i in range(3):
        if keys[i] != 'N': # None
            for index in seg_index[keys[i]]:
                label_mask += np.where(mask==index,1.0,0.0)

            label_mask = cv2.dilate(label_mask, kernel, iterations=5)
            masks[i] = label_mask
    return masks

# 
@torch.no_grad()
def my_morphed_images(
    original_image, reference_images, references, masks, shift_values, interpolation=8, save_dir=None
):

    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(
        original_image, size=(train_args.size, train_args.size)
    )
    original_image = (original_image - 0.5) * 2

    reference_images = torch.stack(reference_images)
    reference_images = F.interpolate(
        reference_images, size=(train_args.size, train_args.size)
    )
    reference_images = (reference_images - 0.5) * 2

    masks = masks[: len(references)]
    masks = torch.from_numpy(np.stack(masks))

    original_image, reference_images, masks = (
        original_image.to(device),
        reference_images.to(device),
        masks.to(device),
    )

    mixed = model(original_image, reference_images, masks, shift_values).cpu()
    mixed = np.asarray(
        np.clip(mixed * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8
    ).transpose(
        (0, 2, 3, 1)
    )  # 0~255

    return mixed

# load image / resize image / get mask of references image / 
def data_preprocess(args):
    original_path = os.path.join(args.original_path,os.listdir(args.original_path)[0])
    references_path = [os.path.join(args.reference_path,os.listdir(args.reference_path)[0])]

    original_image = Image.open(original_path).resize((train_args.size, train_args.size))
    reference_image = [TF.to_tensor(Image.open(references_path[0]).resize((train_args.size, train_args.size)))]
            
    os.system(f'cd face_parsing; python test.py --input ../{references_path[0]}')

    mask = Image.open(os.path.join(args.seg_path,os.listdir(args.seg_path)[0]))
    
    return original_image, reference_image, references_path, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpolation_step", type=int, default=3)
    # parser.add_argument("--ckpt", type=str, default='expr/checkpoints/030000.pt')
    parser.add_argument("--ckpt", type=str, default='expr/checkpoints/celeba_hq_8x8_20M_revised.pt')
    parser.add_argument("--save_path", type=str, default='./result/')
    parser.add_argument("--original_path", type=str, default='images/original/')
    parser.add_argument("--reference_path", type=str, default='images/references/')
    parser.add_argument("--seg_path", type=str, default='images/segmentation/')
    parser.add_argument("--key",nargs='+', type=str, default='mne', help="types:eye,nose,mouth (ex:--want_edit eye")


    args = parser.parse_args()

    device = "cuda"
    ckpt = torch.load(args.ckpt)

    train_args = ckpt["train_args"]
    print("train_args: ", train_args)

    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()

    original_image, reference_images, references_path, mask = data_preprocess(args)

    shift_values = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    masks = mask2masks(mask,args.key)

    generated_images = my_morphed_images(
        original_image,
        reference_images,
        references_path,
        masks,
        shift_values,
        interpolation=args.interpolation_step,
        save_dir=args.save_path,
    )

    for i in range(args.interpolation_step):
        path = f"{args.save_path}/{str(i).zfill(3)}.png"
        Image.fromarray(generated_images[i]).save(path)