"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import argparse
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from training import lpips
from training.model import Generator, Discriminator, Encoder
from training.dataset_ddp import MultiResolutionDataset
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    # initialize the process group
    # nccl : https://tutorials.pytorch.kr/intermediate/dist_tuto.html#communication-backends 참조
    # CUDA tensor들의 집합 연산에 최적화된 백엔드
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 관련 설명 : https://tutorials.pytorch.kr/intermediate/dist_tuto.html
# params : model의 parameter들
# model의 parameter
# op = One of the values from torch.distributed.ReduceOp enum. Specifies an operation used for element-wise reductions
# dist.all_reduce : param.grad.data의 모든 tensor인자들의 합을 알아낸다.
# 평균으로 만든다?
def gather_grad(params, world_size):
    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)

# model을 학습하도록 설정
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# model1의 parameter값을 조절한다
# model1 parameter = model1_parameter*decay + model2_parameter*(1-decay)
def accumulate(model1, model2, decay=0.999):
    with torch.no_grad():
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            # add_ : https://github.com/pytorch/pytorch/issues/23786
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

# 왜 이렇게 하는가?
# https://jangjy.tistory.com/318
# state_dict() : 각 계층을 매개변수 텐서로 매핑되는 python dict 객체
# named_parameters() : model의 (named,parameter) 를 반환한다.
# (https://soundprovider.tistory.com/entry/pytorch-torch%EC%97%90%EC%84%9C-parameter-%EC%A0%91%EA%B7%BC%ED%95%98%EA%B8%B0)
def copy_norm_params(model_tgt, model_src):
    with torch.no_grad():
        src_state_dict = model_src.state_dict()
        tgt_state_dict = model_tgt.state_dict()
        names = [name for name, _ in model_tgt.named_parameters()]
        # model_tgt의 layer이름을 기록하여 중복되는 이름의 src layer를 삭제한다.
        for n in names:
            del src_state_dict[n]
        # model_tgt에 model_src 값을 업데이트 한다.(뭔가 src가 더 많은 layer를 가지고 있는 듯)
        tgt_state_dict.update(src_state_dict)
        model_tgt.load_state_dict(tgt_state_dict)

# Q.??
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_channel_size, device):
    return torch.randn(batch, latent_channel_size, device=device)

# 
class DDPModel(nn.Module):
    def __init__(self, device, args):
        super(DDPModel, self).__init__()

        self.generator = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )
        # g_ema : https://study-grow.tistory.com/entry/gema-EMA-%EA%B5%AC%ED%95%98%EB%8A%94-%EA%B3%B5%EC%8B%9D
        self.g_ema = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
        )

        self.discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier
        )
        
        self.encoder = Encoder(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            channel_multiplier=args.channel_multiplier,
        )

        self.l1_loss = nn.L1Loss(size_average=True)
        self.mse_loss = nn.MSELoss(size_average=True)
        self.e_ema = Encoder(
            args.size,
            args.latent_channel_size,
            args.latent_spatial_size,
            channel_multiplier=args.channel_multiplier,
        )
        self.percept = lpips.exportPerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )

        self.device = device
        self.args = args

    def forward(self, real_img, mode):
        if mode == "G":
            z = make_noise(
                self.args.batch_per_gpu,
                self.args.latent_channel_size,
                self.device,
            )

            fake_img, stylecode = self.generator(z, return_stylecode=True)
            fake_pred = self.discriminator(fake_img)
            adv_loss = g_nonsaturating_loss(fake_pred)
            fake_img = fake_img.detach()
            stylecode = stylecode.detach()
            fake_stylecode = self.encoder(fake_img)
            w_rec_loss = self.mse_loss(stylecode, fake_stylecode)

            return adv_loss, w_rec_loss, stylecode

        elif mode == "D":
            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_img, _ = self.generator(z)
                fake_stylecode = self.encoder(real_img)
                fake_img_from_E, _ = self.generator(
                    fake_stylecode, input_is_stylecode=True
                )

            real_pred = self.discriminator(real_img)
            fake_pred = self.discriminator(fake_img)
            d_loss = d_logistic_loss(real_pred, fake_pred)
            fake_pred_from_E = self.discriminator(fake_img_from_E)
            indomainGAN_D_loss = F.softplus(fake_pred_from_E).mean()

            return (
                d_loss,
                indomainGAN_D_loss,
                real_pred.mean(),
                fake_pred.mean(),
            )

        elif mode == "D_reg":
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss

        elif mode == "E_x_rec":
            fake_stylecode = self.encoder(real_img)
            fake_img, _ = self.generator(fake_stylecode, input_is_stylecode=True)
            x_rec_loss = self.mse_loss(real_img, fake_img)
            perceptual_loss = self.percept(real_img, fake_img).mean()
            fake_pred_from_E = self.discriminator(fake_img)
            indomainGAN_E_loss = F.softplus(-fake_pred_from_E).mean()

            return x_rec_loss, perceptual_loss, indomainGAN_E_loss

        elif mode == "cal_mse_lpips":
            fake_stylecode = self.e_ema(real_img)
            fake_img, _ = self.g_ema(fake_stylecode, input_is_stylecode=True)
            x_rec_loss = self.mse_loss(real_img, fake_img)
            perceptual_loss = self.percept(real_img, fake_img).mean()

            return x_rec_loss, perceptual_loss


def run(ddp_fn, world_size, args):
    print("world size", world_size)

    mp.spawn(ddp_fn, args=(world_size, args), nprocs=world_size, join=True) 


def ddp_main(rank, world_size, args):
    print(f"Running DDP model on rank {rank}.")
    setup(rank, world_size) # env setup
    map_location = f"cuda:{rank}" # rank 번째에 올려놓는다.
    torch.cuda.set_device(map_location)

    
    if args.ckpt: 
        """
        load .pt file
        args를 load한 .pt에 맞게 새롭게 대입한다.
        args.ckpt로 load 여부를 확인한다.
        """
        ckpt = torch.load(args.ckpt, map_location=map_location)
        train_args = ckpt["train_args"]
        print("load model:", args.ckpt)
        train_args.start_iter = int(args.ckpt.split("/")[-1].replace(".pt", ""))
        print(f"continue training from {train_args.start_iter} iter")
        args = train_args
        args.ckpt = True
        
    else:
        args.start_iter = 0

    # create model and move it to GPU with id rank
    model = DDPModel(device=map_location, args=args).to(map_location)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True) # DDP를 가지고 병렬 학습을 가능하게 한다.
    model.train()

    # define generator
    g_module = model.module.generator
    g_ema_module = model.module.g_ema
    g_ema_module.eval()
    accumulate(g_ema_module, g_module, 0)

    # define encoder
    e_module = model.module.encoder
    e_ema_module = model.module.e_ema
    e_ema_module.eval()
    accumulate(e_ema_module, e_module, 0)

    # d_reg의 의미 ??
    # 16/17
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    # define Optimizer
    ## generator Optimizer
    g_optim = optim.Adam(
        g_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    ## discriminator Optimizer
    d_optim = optim.Adam(
        model.module.discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    ## encoder Optimizer
    e_optim = optim.Adam(
        e_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    accum = 0.999

    # 이어서 학습하기 위해서 model's parameter들 load한다.
    if args.ckpt:
        model.module.generator.load_state_dict(ckpt["generator"])
        model.module.discriminator.load_state_dict(ckpt["discriminator"])
        model.module.g_ema.load_state_dict(ckpt["g_ema"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

        model.module.encoder.load_state_dict(ckpt["encoder"])
        e_optim.load_state_dict(ckpt["e_optim"])
        model.module.e_ema.load_state_dict(ckpt["e_ema"])

        del ckpt  # free GPU memory

    # define transform
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # create save path 
    save_dir = "expr"
    os.makedirs(save_dir, 0o777, exist_ok=True)
    os.makedirs(save_dir + "/checkpoints", 0o777, exist_ok=True)

    # load train & val 's lmdb data file
    # lmdb에 size별로 저장되어 있음. args.size를 통해서 해당 해상도 이미지를 transform 적용하여 가져옴
    train_dataset = MultiResolutionDataset(args.train_lmdb, transform, args.size)
    val_dataset = MultiResolutionDataset(args.val_lmdb, transform, args.size)

    print(f"train_dataset: {len(train_dataset)}, val_dataset: {len(val_dataset)}")

    # 분산처리를 하기 위해서 dataset을 gpu개수만큼 묶음으로 나눈다.
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    # data.DataLoader
        # val_dataset : dataset
        # num_workers : https://m.blog.naver.com/qbxlvnf11/221728476511
        # drop_last : dataset을 만들때 마지막 set를 딱 떨어지게 만들어 준다.(버리는 형식)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_per_gpu,
        drop_last=True,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_per_gpu,
        drop_last=True,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 
    train_loader = sample_data(train_loader)

    # 
    pbar = range(args.start_iter, args.iter)
    # tqdm 관련 사용 법 : https://skillmemory.tistory.com/17
    pbar = tqdm(pbar, initial=args.start_iter, mininterval=1)

    # discriminator를 학습하지 않는다.
    requires_grad(model.module.discriminator, False)
    epoch = -1
    gpu_group = dist.new_group(list(range(args.ngpus)))

    for i in pbar:
        # iter & epoch check
        if i > args.iter:
            print("Done!")
            break
        elif i % (len(train_dataset) // args.batch) == 0:
            epoch += 1
            val_sampler.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            print("epoch: ", epoch)

        # load data and upload GPU
        real_img = next(train_loader)
        real_img = real_img.to(map_location)

        # check loss(input noise data)
            # calcul loss and extract 
            # stylecode : input fake image to encoder and get output value(=stylecode)
        adv_loss, w_rec_loss, stylecode = model(None, "G")
        adv_loss = adv_loss.mean()

        with torch.no_grad():
            latent_std = stylecode.std().mean().item()
            latent_channel_std = stylecode.std(dim=1).mean().item()
            latent_spatial_std = stylecode.std(dim=(2, 3)).mean().item()

        # g_loss vs adv_loss
        g_loss = adv_loss * args.lambda_adv_loss
        g_loss_val = g_loss.item()
        adv_loss_val = adv_loss.item()

        # generator propagation
        g_optim.zero_grad() # set gradient value to zero. before back propagation
        g_loss.backward() # back propagation 
        gather_grad(
            g_module.parameters(), world_size
        )  # Explicitly synchronize Generator parameters. There is a gradient sync bug in G.
        g_optim.step()

        w_rec_loss = w_rec_loss.mean()
        w_rec_loss_val = w_rec_loss.item()

        # encoder propagation
        e_optim.zero_grad()
        (w_rec_loss * args.lambda_w_rec_loss).backward()
        e_optim.step()

        # discriminator propagation
        requires_grad(model.module.discriminator, True)
        # D adv
        d_loss, indomainGAN_D_loss, real_score, fake_score = model(real_img, "D")
        d_loss = d_loss.mean()
        indomainGAN_D_loss = indomainGAN_D_loss.mean()
        indomainGAN_D_loss_val = indomainGAN_D_loss.item()

        d_loss_val = d_loss.item()

        d_optim.zero_grad()
        # loss 계산
        (
            d_loss * args.lambda_d_loss
            + indomainGAN_D_loss * args.lambda_indomainGAN_D_loss
        ).backward()
        d_optim.step()

        real_score_val = real_score.mean().item()
        fake_score_val = fake_score.mean().item()

        #---------
        # D reg
        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            d_reg_loss, r1_loss = model(real_img, "D_reg")
            d_reg_loss = d_reg_loss.mean()
            d_optim.zero_grad()
            d_reg_loss.backward()
            d_optim.step()
            r1_val = r1_loss.mean().item()

        # 다시 discriminator 학습 막아 놓는다.
        requires_grad(model.module.discriminator, False)

        # E_x_rec
        x_rec_loss, perceptual_loss, indomainGAN_E_loss = model(real_img, "E_x_rec")
        x_rec_loss = x_rec_loss.mean()
        perceptual_loss = perceptual_loss.mean()

        if indomainGAN_E_loss is not None:
            indomainGAN_E_loss = indomainGAN_E_loss.mean()
            indomainGAN_E_loss_val = indomainGAN_E_loss.item()
        else:
            indomainGAN_E_loss = 0
            indomainGAN_E_loss_val = 0

        # propagate encoder & generator
        e_optim.zero_grad()
        g_optim.zero_grad()

        encoder_loss = (
            x_rec_loss * args.lambda_x_rec_loss
            + perceptual_loss * args.lambda_perceptual_loss
            + indomainGAN_E_loss * args.lambda_indomainGAN_E_loss
        )

        encoder_loss.backward()
        e_optim.step()
        g_optim.step()

        x_rec_loss_val = x_rec_loss.item()
        perceptual_loss_val = perceptual_loss.item()

        pbar.set_description(
            (f"g: {g_loss_val:.4f}; d: {d_loss_val:.4f}; r1: {r1_val:.4f};")
        )

        with torch.no_grad(): # grad 값이 필요 없는 tensor를 만든다.
            accumulate(g_ema_module, g_module, accum)
            accumulate(e_ema_module, e_module, accum)
            # 테스트 할때 val에서 하나 가져온다.
            if i % args.save_network_interval == 0:
                # 새롭게 만들지 않는다?
                copy_norm_params(g_ema_module, g_module)
                copy_norm_params(e_ema_module, e_module)
                x_rec_loss_avg, perceptual_loss_avg = 0, 0
                iter_num = 0

                for test_image in tqdm(val_loader):
                    test_image = test_image.to(map_location) # 이런데 model 
                    x_rec_loss, perceptual_loss = model(test_image, "cal_mse_lpips")
                    x_rec_loss_avg += x_rec_loss.mean()
                    perceptual_loss_avg += perceptual_loss.mean()
                    iter_num += 1

                x_rec_loss_avg /= iter_num
                perceptual_loss_avg /= iter_num

                dist.reduce(
                    x_rec_loss_avg, dst=0, op=dist.ReduceOp.SUM, group=gpu_group
                )

                dist.reduce(
                    perceptual_loss_avg,
                    dst=0,
                    op=dist.ReduceOp.SUM,
                    group=gpu_group,
                )

                if rank == 0:
                    x_rec_loss_avg = x_rec_loss_avg / args.ngpus
                    perceptual_loss_avg = perceptual_loss_avg / args.ngpus
                    x_rec_loss_avg_val = x_rec_loss_avg.item()
                    perceptual_loss_avg_val = perceptual_loss_avg.item()

                    print(
                        f"x_rec_loss_avg: {x_rec_loss_avg_val}, perceptual_loss_avg: {perceptual_loss_avg_val}"
                    )

                    print(
                        f"step={i}, epoch={epoch}, x_rec_loss_avg_val={x_rec_loss_avg_val}, perceptual_loss_avg_val={perceptual_loss_avg_val}, d_loss_val={d_loss_val}, indomainGAN_D_loss_val={indomainGAN_D_loss_val}, indomainGAN_E_loss_val={indomainGAN_E_loss_val}, x_rec_loss_val={x_rec_loss_val}, perceptual_loss_val={perceptual_loss_val}, g_loss_val={g_loss_val}, adv_loss_val={adv_loss_val}, w_rec_loss_val={w_rec_loss_val}, r1_val={r1_val}, real_score_val={real_score_val}, fake_score_val={fake_score_val}, latent_std={latent_std}, latent_channel_std={latent_channel_std}, latent_spatial_std={latent_spatial_std}"
                    )

                    # save argument and model's parameter to .pt file
                    torch.save(
                        {
                            "generator": model.module.generator.state_dict(),
                            "discriminator": model.module.discriminator.state_dict(),
                            "encoder": model.module.encoder.state_dict(),
                            "g_ema": g_ema_module.state_dict(),
                            "e_ema": e_ema_module.state_dict(),
                            "train_args": args,
                            "e_optim": e_optim.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                        },
                        f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                    )


if __name__ == "__main__":
    #----------
    # parser
    #----------
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_lmdb", type=str)
    parser.add_argument("--val_lmdb", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="idol",
        choices=[
            "celeba_hq",
            "afhq",
            "ffhq",
            "lsun/church_outdoor",
            "lsun/car",
            "lsun/bedroom",
            "idol"
        ],
    )
    parser.add_argument("--iter", type=int, default=1400000)
    parser.add_argument("--save_network_interval", type=int, default=10000)
    parser.add_argument("--small_generator", action="store_true")
    parser.add_argument("--batch", type=int, default=16, help="total batch sizes")
    parser.add_argument("--size", type=int, choices=[128, 256, 512, 1024], default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_mul", type=float, default=0.01)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent_channel_size", type=int, default=64)
    parser.add_argument("--latent_spatial_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--normalize_mode",
        type=str,
        choices=["LayerNorm", "InstanceNorm2d", "BatchNorm2d", "GroupNorm"],
        default="LayerNorm",
    )
    parser.add_argument("--mapping_layer_num", type=int, default=8)

    parser.add_argument("--lambda_x_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_adv_loss", type=float, default=1)
    parser.add_argument("--lambda_w_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_d_loss", type=float, default=1)
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_D_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_E_loss", type=float, default=1)

    input_args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    print("{} GPUS!".format(ngpus))

    assert input_args.batch % ngpus == 0
    input_args.batch_per_gpu = input_args.batch // ngpus # setting gpu per batch image 
    input_args.ngpus = ngpus # 이렇게 추가할 수 있음.
    print("{} batch per gpu!".format(input_args.batch_per_gpu))

    run(ddp_main, ngpus, input_args)
