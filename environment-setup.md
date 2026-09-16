# Environment Setup – YZM 513

> **Last updated:** September 2026
>
> **TL;DR:** You need Python with a few packages installed locally (Weeks 1–4),
> and a Google Colab account for GPU work (Weeks 5+). That's it.
> If anything below doesn't work, come to class and we'll fix it together.

---

## 1. What You Need and When

| Weeks | Environment | GPU? |
|:-----:|-------------|:----:|
| 1–4 | Your laptop (any Python) | No |
| 5–13 | Google Colab (free) | Yes (T4, provided) |
| 14–15 | Project: your choice | Depends |

You do **not** need to own a GPU for this course.

---

## 2. Local Setup (Weeks 1–4)

### 2.1 Python

Use whatever Python you already have. Any Python 3 that came with your OS,
Anaconda, a previous course, or a package manager is fine.

Check:

```bash
python --version
# or
python3 --version
```

If you get a version number (3.x), you're good. If you get "command not
found," install Python from [python.org](https://www.python.org/downloads/)
or via your OS package manager, then come back.

> **That's the only requirement.** No specific version is mandated.
> If something breaks due to a very old Python (< 3.9), we'll sort it out
> in class.

### 2.2 Install packages

```bash
pip install numpy numba line_profiler py-spy jupyter matplotlib
```

Later in the semester (Week 9+), you'll also need PyTorch:

```bash
pip install torch torchvision
```

> If `pip` doesn't work, try `pip3` or `python -m pip`.

### 2.3 Verify

```python
import numpy as np
import numba

print(f"NumPy:  {np.__version__}")
print(f"Numba:  {numba.__version__}")

a = np.random.rand(500, 500)
b = np.random.rand(500, 500)
c = a @ b
print("All good ✓")
```

No errors → you're ready for Week 1.

### 2.4 Editor

Use whatever you like: VS Code, PyCharm, Jupyter, Vim, Notepad++.
The course doesn't depend on any specific editor.

---

## 3. Google Colab (Weeks 5 onward)

### 3.1 Getting started

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with a Google account.
3. **File → New notebook.**

Python, NumPy, PyTorch, and most ML libraries are already installed.
No setup needed.

### 3.2 Enabling the GPU

1. **Runtime → Change runtime type**
2. Hardware accelerator → **T4 GPU**
3. Click **Save**
4. Verify:

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # Tesla T4
```

### 3.3 Limits to be aware of

| Limit | Value |
|-------|-------|
| Idle timeout | ~90 min (VM shuts down if you don't interact) |
| Max session | ~12 hours |
| GPU quota | A few hours per day on free tier |
| VRAM | 16 GB (T4) |

**Practical advice:** save your work often. Don't leave a GPU runtime
idle while you go to lunch.

---

## 4. Colab Tips

### 4.1 Running shell commands

Prefix with `!` in any code cell:

```python
!python --version
!nvidia-smi
!ls
```

### 4.2 Installing packages

```python
!pip install cupy-cuda12x
# or (slightly safer, installs into the correct env):
%pip install numba
```

**Packages we'll install during the course:**

```python
!pip install cupy-cuda12x          # Weeks 5–6
!pip install deepspeed             # Week 11
```

> ⚠️ Colab resets when the runtime restarts. Put your `!pip install`
> lines in the **first cell** of every notebook so you can re-run them
> quickly.

### 4.3 Opening a terminal

**Left sidebar → click the `>_` (terminal) icon.**

Or from a menu: **Tools → Command line**.

This gives you an interactive bash shell. Useful for:

```bash
python -m cProfile -s cumulative my_script.py
kernprof -l -v my_script.py
```

### 4.4 Mounting Google Drive (persistence)

Colab's filesystem disappears when the runtime restarts. To keep files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

First time: click the link, authorise, paste the code.

Your files live at `/content/drive/MyDrive/`.

```python
# Example: save a trained model
torch.save(model.state_dict(), '/content/drive/MyDrive/yzm513/model.pt')
```

> Create a folder on your Drive: `MyDrive/yzm513/`

### 4.5 Saving your work

| Method | How |
|--------|-----|
| Save to Drive | File → Save a copy in Drive |
| Download | File → Download .ipynb |
| Push to GitHub | File → Save a copy in GitHub |
| Persistent files | Mount Drive (§4.4) |

### 4.6 Uploading files

```python
# Small files: file picker dialog
from google.colab import files
uploaded = files.upload()

# From a URL
!wget https://example.com/data.csv

# From your Drive (after mounting)
!cp /content/drive/MyDrive/yzm513/data/train.csv .
```

### 4.7 Checking GPU status

```python
!nvidia-smi
```

or in Python:

```python
import torch
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### 4.8 Resetting if something breaks

- **Runtime → Restart runtime** (clears memory, keeps your code cells)
- **Runtime → Factory reset runtime** (full wipe, fresh VM)

Then re-run your imports and `drive.mount(...)`.

---

## 5. Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` not found, but `python3` works | Use `python3` or `pip3`. Or alias: `alias python=python3` |
| `pip install` permission error | Try `pip install --user <pkg>` |
| Colab says "CUDA not available" | Runtime → Change runtime type → T4 GPU. Then restart. |
| Package gone after Colab restart | Re-run your `!pip install` cell. (Colab is ephemeral.) |
| `nvidia-smi` shows a different GPU | Normal. Colab assigns GPUs dynamically. All course code works on any NVIDIA GPU. |
| `nvcc` not found in Colab | `!apt-get install -y nvidia-cuda-toolkit` |
| Something else | Post on the course GitHub Issues or bring it to class. |

---

## 6. Folder Structure (course repo)


Clone:

```bash
git clone https://github.com/adaskin/hpp-for-ai.git
```

---

## 7. Quick Reference

| Task | Command |
|------|---------|
| Check Python | `python --version` |
| Install packages | `pip install numpy numba` |
| Run a script | `python my_script.py` |
| Profile | `python -m cProfile -s cumulative my_script.py` |
| Line profile | `kernprof -l -v my_script.py` |
| Colab: install | `!pip install <pkg>` |
| Colab: shell | `!<command>` |
| Colab: GPU check | `!nvidia-smi` |
| Colab: mount Drive | `from google.colab import drive; drive.mount('/content/drive')` |
| Colab: terminal | Sidebar → `>_` icon |



**Optional addition – a Colab badge:**

You can add an "Open in Colab" badge to any notebook in your repo by adding this at the top of the notebook's first Markdown cell:

```markdown
<a target="_blank" href="https://colab.research.google.com/github/.../.../....ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
```

