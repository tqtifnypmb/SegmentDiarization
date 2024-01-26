from segment import model, SegmentDiarization
import torch
import os
from data_loader import DataLoader
from custom_whisper import load_model
from contextlib import nullcontext
import math
import inspect
from typing import Dict, Iterable, Optional

# -----------------------------------------------------------------------------

# --- performance (best val loss) --
# AMI: 1.02
# Vox Dev: 0.2393
#

# --- ami ---
# test_iters = 50
# eval_iters = 8
# eval_interval = 20
# warmup_iters = 100
# max_iters = 4000 # total number of training iterations

# gradient_accumulation_steps = 3
# batch_size = 2
# learning_rate = 3e-4
# weight_decay = 1e-1
# decay_lr = True
# min_lr = 3e-5
# lr_decay_iters = 4000
# -----------

# --- vox dev ---
# test_iters = 50
# eval_iters = 8
# eval_interval = 10
# warmup_iters = 60
# max_iters = 1000 # total number of training iterations

# gradient_accumulation_steps = 3
# batch_size = 2
# learning_rate = 3e-4
# weight_decay = 1e-1
# decay_lr = True
# min_lr = 3e-5
# lr_decay_iters = 1000
# -----------

# --- vox test ---
test_iters = 80
eval_iters = 8
eval_interval = 20
warmup_iters = 60
max_iters = 2200 # total number of training iterations

gradient_accumulation_steps = 3
batch_size = 2
learning_rate = 6e-4
weight_decay = 1e-1
decay_lr = True
min_lr = 6e-5
lr_decay_iters = 2200
# -----------

n_head = 6
n_layer = 4

device = torch.device("mps") if torch.backends.mps.is_available() else 'cpu'

beta1 = 0.9
beta2 = 0.95

always_save_checkpoint = False
out_dir = "./out"

# ------------------

ctx = nullcontext()
# dataLoader = DataLoader("./ami/audios", "./ami/labels")
# dataLoader = DataLoader("./vox/audios", "./vox/labels")
dataLoader = DataLoader("./vox_test/audios", "./vox_test/labels")

def get_batch(split):
    if split == 'train':
        data = dataLoader.train
    elif split == 'val':
        data = dataLoader.validate
    elif split == 'test':
        data = dataLoader.test

    ix = torch.randint(len(data) - batch_size, (batch_size,))
    audios = []
    labels = []
    for idx in ix:
        label = dataLoader.load_labels(data[idx])
        if label is None:
            return get_batch(split)
        else:
            audio = dataLoader.load_audio(data[idx])
            audios.append(audio)
            labels.append(label)

    x = torch.stack(audios)
    y = torch.concat(labels)
    return x.to(device), y.to(device)
    
@torch.no_grad()
def estimate_loss(raw_model, base_model):
    out = {}
    raw_model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            audios, labels = get_batch(split)
            feature = base_model.encoder(audios)
            with ctx:
                logits, loss = raw_model(feature, labels)
            losses[k] = loss.item()
        out[split] = losses.mean()
    raw_model.train()
    return out

@torch.no_grad()
def estimate_test_loss():
    base_model = load_model("tiny").eval()
    base_model.encoder.eval().to(device)

    raw_model = model.SegmentDecoder(n_ctx=dataLoader.num_of_classes, n_state=base_model.dims.n_audio_state, n_head=n_head, n_layer=n_layer)
    raw_model.to(device)

    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    raw_model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

    checkpoint = None

    losses = torch.zeros(test_iters)
    for idx in range(test_iters):
        with ctx:
            audios, labels = get_batch('test')
            features = base_model.encoder(audios)
            logits, loss = raw_model(features, labels)
            losses[idx] = loss.item()
    print(f"step {iter_num}: test loss {losses.mean():.4f}, best val loss {best_val_loss:.4f}")    

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type) -> torch.optim.AdamW:
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


def train(init_from):
    base_model = load_model("tiny").eval()
    base_model.encoder.eval().to(device)

    raw_model = model.SegmentDecoder(n_ctx=dataLoader.num_of_classes, n_state=base_model.dims.n_audio_state, n_head=n_head, n_layer=n_layer)
    raw_model.to(device)

    optimizer: torch.optim.AdamW = configure_optimizers(raw_model, weight_decay, learning_rate, (beta1, beta2), device)
    iter_num = 0
    best_val_loss = 1e9
    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)

        raw_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss(raw_model, base_model)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
           
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt')) 

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                audios, labels = get_batch('train')
                features = base_model.encoder(audios)
                logits, loss = raw_model(features, labels)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                loss.backward()

        optimizer.step()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, 'ckpt_end_of_iters.pt')) 

def decode():
    base_model = load_model("tiny").eval()
    base_model.encoder.eval().to(device)

    raw_model = model.SegmentDecoder(n_ctx=dataLoader.num_of_classes, n_state=384, n_head=n_head, n_layer=n_layer)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path)

    raw_model.load_state_dict(checkpoint['model'])
    raw_model = raw_model.to(device)
    audios, labels = get_batch('test')
    features = base_model.encoder(audios)
    logits, _ = raw_model(features)
    
    probs = torch.softmax(logits, -1)
    probs = torch.argmax(probs, -1)

    segments: Dict[int, list[Dict]] = {}
    def append_segment(audio_idx: int, segment: Dict):
        if audio_idx in segments:
            segments[audio_idx].append(segment)
        else:
            segments[audio_idx] = [segment]
        

    speaker_changed = (probs[:, :-1] != probs[:, 1:]).nonzero()
    batch_beg_timestep = torch.zeros((features.shape[0],)).to(torch.long)
    for idx in range(speaker_changed.shape[0]):
        audio_idx = speaker_changed[idx][0].item()
        end_timestep = speaker_changed[idx][1].item() + 1
        beg_timestap = batch_beg_timestep[audio_idx].item()

        assert(probs[audio_idx, beg_timestap: end_timestep].allclose(probs[audio_idx, beg_timestap]))

        speaker_id = probs[audio_idx, beg_timestap].item()
        if speaker_id == 0:
            batch_beg_timestep[audio_idx] = end_timestep
            continue

        beg_time = beg_timestap * 20 / 1000
        end_time = end_timestep * 20 / 1000
        ith_speaker = int(speaker_id / (len(dataLoader.speakers) + 1)) + 1
        num_speaker = speaker_id % (len(dataLoader.speakers) + 1) - 1

        append_segment(audio_idx, { "speaker": ith_speaker, "start": beg_time, "end": end_time, "overlap_speakers": num_speaker })
        batch_beg_timestep[audio_idx] = end_timestep

        print(f"speaker: {ith_speaker}, beg: {beg_time}, end: {end_time}, other_speaker: {num_speaker}")

    for audio_idx in range(batch_beg_timestep.shape[-1]):
        beg_timestap = batch_beg_timestep[audio_idx].item()
        if beg_timestap >= 1500:
            continue

        end_timestep = 1500
        assert(probs[audio_idx, beg_timestap: end_timestep].allclose(probs[audio_idx, beg_timestap]))

        speaker_id = probs[audio_idx, beg_timestap].item()
        if speaker_id == 0:
            # silent
            continue

        beg_time = beg_timestap * 20 / 1000
        end_time = end_timestep * 20 / 1000
        ith_speaker = int(speaker_id / (len(dataLoader.speakers) + 1)) + 1
        num_speaker = speaker_id % (len(dataLoader.speakers) + 1) - 1
        print(f"speaker: {ith_speaker}, beg: {beg_time}, end: {end_time}, other_speaker: {num_speaker}")
        append_segment(audio_idx, { "speaker": ith_speaker, "start": beg_time, "end": end_time, "overlap_speakers": num_speaker })

def release():
    raw_model = model.SegmentDecoder(n_ctx=dataLoader.num_of_classes, n_state=384, n_head=n_head, n_layer=n_layer)
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path)

    raw_model.load_state_dict(checkpoint['model'])
    dims = SegmentDiarization.ModelDimensions(384, n_head, n_layer, dataLoader.num_of_classes, len(dataLoader.speakers))
    checkpoint = {
        'model': raw_model.state_dict(),
        'dims': dims
    }
    torch.save(checkpoint, os.path.join(out_dir, 'release_ckpt.pt')) 

def test():
    model = SegmentDiarization.SegmentDiarization("./out/release_ckpt.pt")
    model.model.to(device)

    base_model = load_model("tiny").eval()
    base_model.encoder.to(device)

    # audios, labels = get_batch('test')
    
    filename = dataLoader.test[40]
    print(filename)
    audios = dataLoader.load_audio(filename)
    audios = audios.to(device)
    features = base_model.encoder(audios.unsqueeze(0))
    model.decode(features)

if __name__ == '__main__':
    # train('resume')
    # decode()
    # estimate_test_loss()
    test()