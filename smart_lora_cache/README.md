# Smart LoRA Cache — reForge / A1111 Extension

Replaces the default **FIFO** LoRA memory eviction with a **batch-aware, frequency-optimal** strategy. When you run a batch of images with many different LoRAs, this extension dramatically reduces redundant disk loads.

---

## The problem it solves

The built-in `lora_in_memory_limit` setting keeps N LoRA files in RAM between generations. But the eviction policy is naive FIFO — it evicts whatever was loaded first, with no awareness of what's coming next in the batch. With 20 images using 30 different LoRAs and a cache of 10, you can end up loading the same LoRA from disk multiple times.

---

## What this extension does

### 1. Pre-resolves all prompts before generation starts

Before image 1 is generated, the extension:
- Reads all prompts in the batch (including `all_prompts` lists from X/Y/Z grids etc.)
- Runs **sd-dynamic-prompts** wildcard resolution on each prompt (using the correct seed per image)
- Scans every resolved prompt for `<lora:name:...>` tokens

> ⚠️ If you use custom wildcard extensions that modify prompts *after* sd-dynamic-prompts resolution, those LoRAs won't be visible to the pre-scan. They are handled as graceful cache misses with LFU-remaining eviction as a fallback.

### 2. Pre-warms the cache with the most-used LoRAs

After scanning, the extension loads the **top-N most frequently used LoRAs** (where N = your `lora_in_memory_limit` setting) into memory at strength 0 before the batch starts. The most-reused LoRAs are never cold-loaded during generation.

### 3. LFU-remaining eviction for mid-batch cache misses

When a LoRA isn't in cache and the cache is full, instead of evicting the oldest (FIFO), the extension evicts the cached LoRA that appears **least often in the remaining images**. This is the optimal Bélády cache replacement policy, possible because we know the full batch upfront.

Ties are broken by LRU (oldest insertion order).

---

## Installation

1. Copy the `smart-lora-cache/` folder into your `extensions/` directory:
   ```
   <webui root>/extensions/smart-lora-cache/
   ```
2. Restart the webui.
3. The extension will appear as a collapsible accordion in txt2img and img2img.

---

## Settings

The **cache size** is still controlled by the existing webui setting:
> **Settings → LoRA → Number of Lora networks to keep cached in memory**

Set it to 0 to disable caching entirely (this extension does nothing in that case).

Within the extension accordion you can:
- **Enable/disable** the smart cache per-session
- **Enable verbose logging** to see per-LoRA frequency counts and eviction decisions in the console

---

## Compatibility

| Feature | Status |
|---|---|
| sd-dynamic-prompts wildcards | ✅ Full support (pre-resolved) |
| A1111 built-in `__wildcard__` syntax | ✅ Resolved via sd-dynamic-prompts if installed |
| Custom post-resolution wildcard extensions | ⚠️ Best-effort (LFU-remaining fallback) |
| X/Y/Z grid | ✅ (uses `all_prompts` list) |
| batch_size > 1 | ✅ |
| n_iter > 1 | ✅ |
| reForge | ✅ Primary target |
| A1111 | ✅ Should work |
| Forge | ✅ Should work |

---

## How the eviction math works

Given cache size N and a batch of M images:

- **Pre-warm phase**: rank all LoRAs by frequency across M images → cache top-N
- **Miss phase**: need to load LoRA X, cache is full → evict LoRA Y where Y has the lowest count in images `[current+1 .. M]`

This is equivalent to Bélády's algorithm and is provably optimal when the future access sequence is known — which it is, because we pre-resolved everything.

---

## Example

**Batch**: 20 images, 30 unique LoRAs, cache limit = 10

| Strategy | Estimated disk loads |
|---|---|
| Default FIFO | ~30–45 (many re-loads) |
| Smart (this extension) | ~20–22 (near-optimal) |

Exact savings depend on your prompt distribution.
