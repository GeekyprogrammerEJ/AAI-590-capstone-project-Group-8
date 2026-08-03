# Running the QLoRA fine-tune on a rented GPU (SSH)

`src/train_qlora.py` is **self-contained** — it downloads MedMCQA + MedQA from the
Hugging Face Hub itself, so you only need that one file on the GPU host (no repo
clone, no GitHub push required). Any provider that gives you SSH + a CUDA GPU
works (RunPod, Lambda, Vast.ai, Paperspace). RunPod is used as the concrete
example below.

A single **24 GB** card (RTX 4090 / A5000 / L4) runs the defaults comfortably;
a **16 GB** card (T4 / A4000) works with the smaller settings noted in step 5.

---

## 1. Launch a GPU pod
- RunPod → **Deploy** → **Community Cloud** → pick a 24 GB GPU (e.g. RTX 4090,
  ~$0.3–0.5/hr).
- Template: **RunPod PyTorch** (ships CUDA + PyTorch).
- Enable **SSH** and add your public key (`~/.ssh/id_ed25519.pub`) in RunPod
  → Settings → SSH Public Keys.
- After it boots, copy the SSH command it shows, e.g.
  `ssh root@194.x.x.x -p 40123 -i ~/.ssh/id_ed25519`

## 2. Copy the training script to the host
Run this **on your Mac** (note the capital `-P` for the port, and the quotes
around the path because it contains spaces):
```bash
scp -P 40123 \
  "/Volumes/Extreme SSD/final project/capstone-medqa/src/train_qlora.py" \
  root@194.x.x.x:~/
```

## 3. SSH in and check the GPU
```bash
ssh root@194.x.x.x -p 40123 -i ~/.ssh/id_ed25519
nvidia-smi          # confirm the GPU is visible
```

## 4. Install the extra dependencies
The PyTorch template already has torch; add the rest (no version pins, so numpy
is left alone):
```bash
pip install -U transformers datasets peft bitsandbytes accelerate
```

## 5. (16 GB cards only) shrink the settings
Edit the `CFG` block near the top of `train_qlora.py`:
```python
n_train=6000, max_seq_len=512, batch_size=1, grad_accum=16,
```
On a 24 GB card the defaults are fine.

## 6. Authenticate for the base model
Mistral is **gated** — accept its license at
`huggingface.co/mistralai/Mistral-7B-Instruct-v0.2`, then:
```bash
huggingface-cli login          # paste a token from hf.co/settings/tokens
```
**Or** skip gating entirely: edit `CFG["model_name"]` to
`"Qwen/Qwen2.5-7B-Instruct"` (ungated, comparable size) — no token needed.

## 7. Train (inside tmux, so it survives disconnects)
```bash
tmux new -s train
python train_qlora.py 2>&1 | tee train.log
# detach: Ctrl-b then d      reattach later: tmux attach -t train
```
You should see the trainable-parameter count (~0.5–1% of the model), then a
decreasing loss, and finally a validation-accuracy line.

## 8. Retrieve the trained adapter
The adapter is small (tens of MB). **On your Mac:**
```bash
scp -P 40123 -r root@194.x.x.x:~/qlora-medqa-adapter \
  "/Volumes/Extreme SSD/final project/capstone-medqa/"
```
(Alternatively, `model.push_to_hub("<user>/qlora-medqa")` from the pod.)

## 9. Stop billing
**Terminate / stop the pod** in the RunPod dashboard when done — you are billed
while it runs.

---

### Getting the baseline (zero-shot) number for the Results section
Run the evaluation on the **base model before training** to get the comparison
point. The quickest way: in a Python shell on the pod,
```python
from train_qlora import build_model_and_tokenizer, load_records, evaluate, CFG
# temporarily point build_model_and_tokenizer at the base model without adapters,
# or simply run evaluate() before trainer.train() by editing main().
```
The difference between the base and fine-tuned accuracy is the headline result.

### Troubleshooting
- **`CUDA out of memory`** → apply step 5, and/or lower `max_seq_len`.
- **`gated repo` / 401** → step 6 (accept license + login), or switch to Qwen.
- **bitsandbytes import error** → `pip install -U bitsandbytes`; confirm the pod
  has a real NVIDIA GPU (`nvidia-smi`).
- **Session died** → that is why step 7 uses `tmux`; reattach with
  `tmux attach -t train`.
