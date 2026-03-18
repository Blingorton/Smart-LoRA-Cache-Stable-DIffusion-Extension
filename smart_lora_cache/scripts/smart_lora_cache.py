"""
Smart LoRA Cache — reForge/A1111 extension
==========================================
Replaces the default FIFO LoRA eviction policy with a batch-aware,
frequency-optimal strategy aware of your full extension pipeline:

  wildcard-resolver  (before_process)
  → sd-dynamic-prompts  (process)
  → antiwildcards  (process_before_every_sampling)
  → [sampling]

Two-pass strategy:
  Pass 1 (before_process):              pre-resolve dp wildcards → pre-warm cache
  Pass 2 (process_before_every_sampling): read final prompt after antiwildcards
                                          → correct batch_loras, push to history

"Generate forever" support:
  Rolling frequency history (configurable window) persists across all batches.
  Cache decisions blend current-batch frequency + history for progressively
  smarter decisions over long runs.

Pinned LoRAs:
  LoRAs you always want in cache (e.g. slider LoRAs used at variable strengths).
  They are never evicted. Strength is still applied per-image as normal —
  caching only stores the weight tensors, not the strength value.
"""

import re
import logging
import threading
from collections import defaultdict, OrderedDict, deque
from typing import Dict, List, Optional, Set

import gradio as gr

import modules.scripts as scripts
from modules import shared
from modules.processing import StableDiffusionProcessing

logger = logging.getLogger("smart_lora_cache")
if not logger.handlers:
    logging.basicConfig()

# ---------------------------------------------------------------------------
# sd-dynamic-prompts
# ---------------------------------------------------------------------------
try:
    from dynamicprompts.generators import RandomPromptGenerator
    HAS_DYNAMIC_PROMPTS = True
except ImportError:
    HAS_DYNAMIC_PROMPTS = False

# LoRA token regex:  <lora:name:...>  (name may not start with space or colon)
LORA_RE = re.compile(r"<lora:([^:>\s][^:>]*)(?::[^>]*)?>", re.IGNORECASE)

# wildcard-resolver sentinel
WRC_START = "||WRC||"
WRC_END   = "||/WRC||"

# ---------------------------------------------------------------------------
# Persistence — save/load pinned list and cache limit across restarts
# ---------------------------------------------------------------------------
import json, os as _os

_PERSIST_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)),  # extension root
    "smart_lora_cache_settings.json",
)

def _load_persisted() -> dict:
    try:
        with open(_PERSIST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_persisted(data: dict):
    try:
        with open(_PERSIST_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[SmartLoRACache] Could not save settings: {e}")

# Load pinned list on startup so it survives restarts
_persisted       = _load_persisted()
_startup_pinned  = set(_persisted.get("pinned", []))

# ---------------------------------------------------------------------------
# Rolling history — persists across batches
# ---------------------------------------------------------------------------
_history_lock  = threading.Lock()
_lora_history: deque           = deque()
_history_freq: Dict[str, int]  = defaultdict(int)
_history_window: int           = 200   # updated from UI each batch


def _history_push(lora_set: Set[str]):
    fs = frozenset(lora_set)
    with _history_lock:
        _lora_history.append(fs)
        for name in fs:
            _history_freq[name] += 1
        while len(_lora_history) > _history_window:
            oldest = _lora_history.popleft()
            for name in oldest:
                _history_freq[name] -= 1
                if _history_freq[name] <= 0:
                    _history_freq.pop(name, None)


def _blended_scores(
    batch_freq: Dict[str, int],
    history_weight: float,
) -> Dict[str, float]:
    with _history_lock:
        hist = dict(_history_freq)

    all_loras = set(batch_freq) | set(hist)
    max_batch = max(batch_freq.values(), default=1)
    max_hist  = max(hist.values(), default=1)

    scores: Dict[str, float] = {}
    for name in all_loras:
        b = batch_freq.get(name, 0) / max_batch
        h = hist.get(name, 0)       / max_hist
        scores[name] = (1.0 - history_weight) * b + history_weight * h
    return scores


# ---------------------------------------------------------------------------
# Per-batch state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_state: Dict = {
    "enabled":        False,
    "batch_loras":    [],
    "image_index":    0,
    "history_weight": 0.4,
    "pinned":         _startup_pinned,   # set[str] — persisted across restarts
    "debug_log":      False,
}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _strip_wrc(prompt: str) -> str:
    cs = prompt.find(WRC_START)
    if cs == -1:
        return prompt
    ce = prompt.find(WRC_END, cs)
    if ce == -1:
        return prompt
    return prompt[:cs] + prompt[ce + len(WRC_END):]


def _resolve_one(prompt: str, seed: int) -> str:
    if not HAS_DYNAMIC_PROMPTS:
        return prompt
    try:
        gen     = RandomPromptGenerator(seed=seed)
        results = gen.generate(prompt, count=1)
        return results[0] if results else prompt
    except Exception as e:
        logger.debug(f"[SmartLoRACache] dp resolve (non-fatal): {e}")
        return prompt


def _extract_loras(prompt: str) -> Set[str]:
    return {m.group(1).strip() for m in LORA_RE.finditer(prompt)}


def _prescan_batch(p: StableDiffusionProcessing) -> List[Set[str]]:
    """
    First-pass LoRA scan. Resolves dp wildcards, strips WRC sentinel.
    antiwildcards injections are NOT visible yet — corrected in pass 2.
    """
    all_prompts = getattr(p, "all_prompts", None)
    if all_prompts and isinstance(all_prompts, list) and all_prompts:
        raw = list(all_prompts)
    else:
        prompt     = getattr(p, "prompt", "") or ""
        batch_size = getattr(p, "batch_size", 1) or 1
        n_iter     = getattr(p, "n_iter",    1) or 1
        raw = [prompt] * (batch_size * n_iter)

    raw  = [_strip_wrc(pr) for pr in raw]
    seed = int(getattr(p, "seed", 0) or 0)
    return [_extract_loras(_resolve_one(pr, seed + i)) for i, pr in enumerate(raw)]


# ---------------------------------------------------------------------------
# Cache sizing helpers
# ---------------------------------------------------------------------------

def _get_cache_limit() -> int:
    """Read lora_in_memory_limit from shared.opts (the real setting)."""
    return int(getattr(shared.opts, "lora_in_memory_limit", 0))


def _set_cache_limit(n: int):
    """Write lora_in_memory_limit back to shared.opts so it persists."""
    try:
        shared.opts.lora_in_memory_limit = int(n)
        shared.opts.save(shared.config_filename)
    except Exception as e:
        logger.warning(f"[SmartLoRACache] Could not save lora_in_memory_limit: {e}")


def _estimate_lora_size_mb() -> float:
    """
    Rough heuristic: inspect whatever is in networks_in_memory and average
    the size, or fall back to a reasonable SD1.5/SDXL estimate.
    """
    if _lora_networks_ref is None:
        return 144.0  # ~144 MB is a typical SD1.5 LoRA in fp16

    mem = getattr(_lora_networks_ref, "networks_in_memory", {})
    if not mem:
        return 144.0

    sizes = []
    for net in mem.values():
        try:
            # Sum parameter bytes across all modules in the network
            total = sum(
                p.numel() * p.element_size()
                for m in net.modules.values()
                for p in [getattr(m, "up", None), getattr(m, "down", None),
                           getattr(m, "alpha", None)]
                if p is not None and hasattr(p, "numel")
            )
            if total > 0:
                sizes.append(total / (1024 ** 2))
        except Exception:
            pass

    return sum(sizes) / len(sizes) if sizes else 144.0


def _recommend_cache_size() -> str:
    """
    Suggest a cache limit based on available RAM (not VRAM — LoRA cache
    lives in system RAM by default in reForge).
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
        # Leave at least 4 GB headroom
        usable_mb    = max(0, available_mb - 4096)
        lora_mb      = _estimate_lora_size_mb()
        suggested    = max(1, int(usable_mb / lora_mb))
        suggested    = min(suggested, 50)  # cap at 50 — sanity
        return (
            f"~{available_mb/1024:.1f} GB RAM free · "
            f"~{lora_mb:.0f} MB/LoRA estimated · "
            f"suggested limit: {suggested}"
        )
    except ImportError:
        return "Install psutil for RAM-based recommendations (pip install psutil)"
    except Exception as e:
        return f"Could not estimate: {e}"


# ---------------------------------------------------------------------------
# Cache selection / eviction
# ---------------------------------------------------------------------------

def _choose_prewarm(
    batch_loras: List[Set[str]],
    limit: int,
    history_weight: float,
    pinned: Set[str],
) -> List[str]:
    """
    Select LoRAs to pre-warm. Pinned LoRAs always included first,
    remainder filled by blended score.
    """
    batch_freq: Dict[str, int] = defaultdict(int)
    for s in batch_loras:
        for n in s:
            batch_freq[n] += 1

    scores = _blended_scores(dict(batch_freq), history_weight)

    # Start with pinned (that exist in available networks)
    result: List[str] = [p for p in pinned]
    remaining_slots   = limit - len(result)

    if remaining_slots > 0:
        candidates = [n for n in sorted(scores, key=lambda n: -scores[n])
                      if n not in pinned]
        result.extend(candidates[:remaining_slots])

    return result[:limit]


def _lfu_remaining_evict(
    networks_in_memory: "OrderedDict",
    batch_loras: List[Set[str]],
    current_index: int,
    history_weight: float,
    pinned: Set[str],
) -> Optional[str]:
    """
    Evict least-valuable cached LoRA for remaining images + history.
    Pinned LoRAs are never evicted.
    """
    evictable = [k for k in networks_in_memory if k not in pinned]
    if not evictable:
        # All cached LoRAs are pinned — nothing to evict.
        # Caller will have to let the normal path handle overflow.
        return None

    remaining                  = batch_loras[current_index + 1:]
    future_freq: Dict[str, int] = defaultdict(int)
    for s in remaining:
        for n in s:
            future_freq[n] += 1

    scores = _blended_scores(dict(future_freq), history_weight)
    keys   = list(networks_in_memory.keys())

    return min(
        evictable,
        key=lambda name: (scores.get(name, 0.0), keys.index(name)),
    )


# ---------------------------------------------------------------------------
# Patch
# ---------------------------------------------------------------------------
_patch_installed   = False
_lora_networks_ref = None


def _install_patches():
    global _patch_installed, _lora_networks_ref
    if _patch_installed:
        return

    try:
        import extensions_builtin.Lora.networks as lora_networks  # type: ignore
    except ImportError:
        logger.warning("[SmartLoRACache] Could not import Lora.networks — patch skipped.")
        return

    _lora_networks_ref = lora_networks
    _original_purge    = lora_networks.purge_networks_from_memory

    def smart_purge():
        with _state_lock:
            active      = _state["enabled"]
            batch_loras = list(_state["batch_loras"])
            cur_idx     = _state["image_index"]
            hw          = _state["history_weight"]
            pinned      = set(_state["pinned"])

        if not active:
            _original_purge()
            return

        limit = _get_cache_limit()
        if limit <= 0:
            # Keep pinned even when cache is "off" — they cost nothing extra
            to_remove = [k for k in list(lora_networks.networks_in_memory)
                         if k not in pinned]
            for k in to_remove:
                lora_networks.networks_in_memory.pop(k, None)
            return

        while len(lora_networks.networks_in_memory) > limit:
            victim = _lfu_remaining_evict(
                lora_networks.networks_in_memory, batch_loras, cur_idx, hw, pinned
            )
            if victim is None:
                break
            lora_networks.networks_in_memory.pop(victim, None)
            logger.debug(f"[SmartLoRACache] Evicted '{victim}'")

    lora_networks.purge_networks_from_memory = smart_purge
    _patch_installed = True
    logger.info("[SmartLoRACache] Smart eviction patch installed.")


def _prewarm_cache(to_prewarm: List[str]):
    """
    Populate networks_in_memory directly via load_network() (NOT load_networks()).
    This avoids corrupting reForge's UNet patch hash which is keyed on
    [filename, unet_strength, te_strength].
    """
    if _lora_networks_ref is None:
        return

    limit = _get_cache_limit()
    # Pinned LoRAs should always fit even if limit is low — clamp to
    # at least the pinned count so we don't thrash.
    with _state_lock:
        pinned_count = len(_state["pinned"])
    effective_limit = max(limit, pinned_count)

    available       = getattr(_lora_networks_ref, "available_networks",        {})
    available_alias = getattr(_lora_networks_ref, "available_network_aliases", {})

    loaded = 0
    for name in to_prewarm:
        if len(_lora_networks_ref.networks_in_memory) >= effective_limit:
            break
        if name in _lora_networks_ref.networks_in_memory:
            continue

        net_on_disk = available.get(name) or available_alias.get(name)
        if net_on_disk is None:
            logger.debug(f"[SmartLoRACache] Pre-warm: '{name}' not in available_networks")
            continue

        try:
            network = _lora_networks_ref.load_network(name, net_on_disk)
            if network is not None:
                _lora_networks_ref.networks_in_memory[name] = network
                loaded += 1
                logger.debug(f"[SmartLoRACache] Pre-warmed '{name}'")
        except Exception as e:
            logger.debug(f"[SmartLoRACache] Pre-warm error '{name}': {e}")

    if loaded:
        logger.info(f"[SmartLoRACache] Pre-warmed {loaded} LoRA(s).")


# ---------------------------------------------------------------------------
# Live cache inspector helpers (called from UI buttons)
# ---------------------------------------------------------------------------

def _cache_contents_text() -> str:
    if _lora_networks_ref is None:
        return "Extension not yet initialised — generate at least one image first."

    mem   = getattr(_lora_networks_ref, "networks_in_memory", {})
    limit = _get_cache_limit()

    if not mem:
        return f"Cache empty  (limit: {limit})"

    with _state_lock:
        pinned = set(_state["pinned"])

    lines = [f"Cache: {len(mem)}/{limit} slots used\n"]
    with _history_lock:
        hist = dict(_history_freq)

    for name in mem:
        pin_tag  = " 📌" if name in pinned else ""
        hist_tag = f"  [{hist.get(name, 0)}× in history]" if hist else ""
        lines.append(f"  ✓ {name}{pin_tag}{hist_tag}")
    return "\n".join(lines)


def _auto_pin_candidates(threshold_pct: float) -> List[str]:
    """
    Return LoRA names that appear in >= threshold_pct % of history images.
    threshold_pct is 0–100.
    """
    with _history_lock:
        total = len(_lora_history)
        freq  = dict(_history_freq)

    if total == 0:
        return []

    cutoff = total * (threshold_pct / 100.0)
    return sorted(
        [name for name, count in freq.items() if count >= cutoff],
        key=lambda n: -freq[n],
    )


def _merge_pinned_text(existing_text: str, new_names: List[str]) -> str:
    """Merge new_names into the existing pinned text box, deduped, sorted."""
    existing = {l.strip() for l in existing_text.splitlines() if l.strip()}
    merged   = sorted(existing | set(new_names))
    return "\n".join(merged)


def _top_history_text(n: int = 15) -> str:
    with _history_lock:
        total  = len(_lora_history)
        top    = sorted(_history_freq.items(), key=lambda x: -x[1])[:n]

    if not top:
        return "No history yet. Generate some images first."

    lines = [f"Top LoRAs across last {total} images:\n"]
    for rank, (name, count) in enumerate(top, 1):
        bar = "█" * min(20, count)
        lines.append(f"  {rank:2}. {bar} {count}×  {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class Script(scripts.Script):

    def title(self):
        return "Smart LoRA Cache"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tab = "i2i" if is_img2img else "t2i"

        with gr.Accordion("🧠 Smart LoRA Cache", open=False):
            enabled = gr.Checkbox(
                label="Enable smart LoRA cache",
                value=True,
                elem_id=f"slc_enabled_{tab}",
            )

            # ── Cache size ────────────────────────────────────────────────
            with gr.Group():
                gr.Markdown("### Cache size")
                with gr.Row():
                    cache_limit = gr.Slider(
                        minimum=0, maximum=50, step=1,
                        value=_get_cache_limit,   # reads live setting on render
                        label="LoRAs to keep in memory  (0 = disabled)",
                        elem_id=f"slc_limit_{tab}",
                    )
                    apply_limit_btn = gr.Button("Apply & save", size="sm", scale=0)

                recommend_box = gr.Textbox(
                    label="", value="", interactive=False, max_lines=2,
                    elem_id=f"slc_recommend_{tab}",
                )
                recommend_btn = gr.Button("💡 Recommend based on free RAM", size="sm")

                gr.Markdown(
                    "_This syncs with **Settings → LoRA → Number of Lora networks "
                    "to keep cached in memory** — you only need to set it in one place._\n\n"
                    "**Why caching helps:** every image that reuses a cached LoRA "
                    "skips a disk read (typically 50–500 ms per LoRA depending on "
                    "drive speed and LoRA size). With 20+ LoRAs per batch the savings "
                    "compound quickly."
                )

            # ── Pinned LoRAs ───────────────────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "### Pinned LoRAs\n"
                    "Always kept in cache, never evicted — regardless of cache size "
                    "or frequency. Ideal for LoRAs you use at variable strengths in "
                    "nearly every image (e.g. contrast/saturation/brightness sliders).\n\n"
                    "**Strength is independent of caching** — the cache stores weight "
                    "tensors only; the strength multiplier is applied fresh each image.\n\n"
                    "_This list is saved to disk and restored on restart._"
                )

                # Manual pin list
                _startup_pinned_text = "\n".join(sorted(_startup_pinned))
                pinned_box = gr.Textbox(
                    label="Pinned LoRA names (one per line, exact filename without .safetensors)",
                    value=_startup_pinned_text,
                    placeholder=(
                        "il_contrast_slider_d1\n"
                        "il_saturation_slider_d1\n"
                        "il_brightness_slider_d1"
                    ),
                    lines=4,
                    elem_id=f"slc_pinned_{tab}",
                )
                with gr.Row():
                    pin_apply_btn  = gr.Button("💾 Apply & save pins", size="sm")
                    pin_status_box = gr.Textbox(
                        label="", value=(
                            f"{len(_startup_pinned)} pin(s) loaded from disk."
                            if _startup_pinned else ""
                        ),
                        interactive=False, max_lines=1,
                        elem_id=f"slc_pin_status_{tab}",
                    )

                # Auto-pin from history
                gr.Markdown(
                    "**Auto-pin from history** — scan your generation history and "
                    "automatically add high-frequency LoRAs to the pinned list. "
                    "Useful after a long 'generate forever' session to discover "
                    "which LoRAs you actually use most."
                )
                with gr.Row():
                    autopin_threshold = gr.Slider(
                        minimum=1, maximum=100, step=1, value=50,
                        label="Auto-pin threshold (% of history images)",
                        info="Pin any LoRA that appears in at least this % of recorded images.",
                        elem_id=f"slc_autopin_thresh_{tab}",
                        scale=3,
                    )
                    autopin_preview_btn = gr.Button("🔍 Preview candidates", size="sm", scale=1)
                    autopin_add_btn     = gr.Button("➕ Add to pinned list", size="sm", scale=1)

                autopin_preview_box = gr.Textbox(
                    label="Candidates at current threshold",
                    value="",
                    interactive=False, lines=3,
                    elem_id=f"slc_autopin_preview_{tab}",
                )

            # ── History / generate-forever ─────────────────────────────────
            with gr.Group():
                gr.Markdown(
                    "### Generate-forever mode\n"
                    "The extension learns which LoRAs you use most over time. "
                    "The **history blend** weight controls how much past usage "
                    "influences cache decisions vs. just the current batch."
                )
                history_weight = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.05, value=0.4,
                    label="History blend  (0 = this batch only · 1 = history only)",
                    info="0.3–0.5 is a good default. Raise if you generate for hours.",
                    elem_id=f"slc_hw_{tab}",
                )
                history_window = gr.Slider(
                    minimum=20, maximum=2000, step=20, value=200,
                    label="History window (images)",
                    info="How many past images to remember. Larger = slower to adapt to style changes.",
                    elem_id=f"slc_win_{tab}",
                )

                with gr.Row():
                    refresh_hist_btn = gr.Button("🔄 Refresh stats", size="sm")
                    clear_hist_btn   = gr.Button("🗑 Clear history",  size="sm")

                history_stats = gr.Textbox(
                    label="Top LoRAs in history",
                    value="No history yet.",
                    interactive=False, lines=8,
                    elem_id=f"slc_hist_{tab}",
                )

            # ── Live cache inspector ───────────────────────────────────────
            with gr.Group():
                gr.Markdown("### Live cache inspector")
                with gr.Row():
                    inspect_btn  = gr.Button("🔍 Show cache contents", size="sm")
                cache_inspector = gr.Textbox(
                    label="",
                    value="Click 'Show cache contents' to inspect.",
                    interactive=False, lines=6,
                    elem_id=f"slc_inspector_{tab}",
                )

            # ── Debug ──────────────────────────────────────────────────────
            debug_log = gr.Checkbox(
                label="Verbose logging (console)", value=False,
                elem_id=f"slc_debug_{tab}",
            )

            # ── Wire up buttons ────────────────────────────────────────────

            def do_apply_limit(n):
                _set_cache_limit(int(n))
                return f"Saved: cache limit = {int(n)}"

            def do_recommend():
                return _recommend_cache_size()

            def do_apply_pins(text):
                names = {l.strip() for l in text.splitlines() if l.strip()}
                with _state_lock:
                    _state["pinned"] = names
                # Persist to disk
                _save_persisted({"pinned": sorted(names)})
                if names:
                    return f"Saved {len(names)} pin(s): {', '.join(sorted(names))}"
                return "Pins cleared and saved."

            def do_autopin_preview(threshold):
                candidates = _auto_pin_candidates(float(threshold))
                if not candidates:
                    with _history_lock:
                        n = len(_lora_history)
                    if n == 0:
                        return "No history yet — generate some images first."
                    return f"No LoRAs appear in ≥{int(threshold)}% of {n} recorded images."
                with _history_lock:
                    freq = dict(_history_freq)
                    n    = len(_lora_history)
                lines = [f"{len(candidates)} candidate(s) from {n} images at ≥{int(threshold)}%:\n"]
                for name in candidates:
                    pct = freq.get(name, 0) / n * 100
                    lines.append(f"  {name}  ({pct:.0f}%)")
                return "\n".join(lines)

            def do_autopin_add(threshold, current_text):
                candidates = _auto_pin_candidates(float(threshold))
                if not candidates:
                    return current_text, "No candidates at this threshold."
                new_text = _merge_pinned_text(current_text, candidates)
                # Also apply immediately
                names = {l.strip() for l in new_text.splitlines() if l.strip()}
                with _state_lock:
                    _state["pinned"] = names
                _save_persisted({"pinned": sorted(names)})
                added = [c for c in candidates
                         if c not in {l.strip() for l in current_text.splitlines()}]
                return new_text, f"Added {len(added)} LoRA(s). Total pinned: {len(names)}."

            def do_refresh_hist():
                return _top_history_text()

            def do_clear_hist():
                with _history_lock:
                    _lora_history.clear()
                    _history_freq.clear()
                return "History cleared."

            def do_inspect():
                return _cache_contents_text()

            apply_limit_btn.click(  do_apply_limit,   [cache_limit],                        [recommend_box])
            recommend_btn.click(    do_recommend,      [],                                   [recommend_box])
            pin_apply_btn.click(    do_apply_pins,     [pinned_box],                         [pin_status_box])
            autopin_preview_btn.click(do_autopin_preview, [autopin_threshold],               [autopin_preview_box])
            autopin_add_btn.click(  do_autopin_add,   [autopin_threshold, pinned_box],       [pinned_box, pin_status_box])
            refresh_hist_btn.click( do_refresh_hist,   [],                                   [history_stats])
            clear_hist_btn.click(   do_clear_hist,     [],                                   [history_stats])
            inspect_btn.click(      do_inspect,        [],                                   [cache_inspector])

        return [enabled, history_weight, history_window, debug_log]

    # ------------------------------------------------------------------
    # HOOK 1 — before_process
    # ------------------------------------------------------------------
    def before_process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        history_weight: float,
        history_window: int,
        debug_log: bool,
    ):
        global _history_window

        with _state_lock:
            _state["enabled"] = False

        if not enabled:
            return

        limit = _get_cache_limit()
        logger.setLevel(logging.DEBUG if debug_log else logging.INFO)
        _history_window = int(history_window)

        _install_patches()

        with _state_lock:
            pinned = set(_state["pinned"])

        prescan  = _prescan_batch(p)
        n_unique = len({n for s in prescan for n in s})
        logger.info(
            f"[SmartLoRACache] Batch: {len(prescan)} images, "
            f"{n_unique} unique LoRAs (post-dp, pre-antiwildcards), "
            f"limit={limit}, hw={history_weight:.2f}, win={_history_window}, "
            f"pinned={sorted(pinned)}"
        )

        if debug_log:
            freq: Dict[str, int] = defaultdict(int)
            for s in prescan:
                for n in s: freq[n] += 1
            for name, count in sorted(freq.items(), key=lambda x: -x[1]):
                logger.debug(f"  [prescan] {name}: {count}×")

        with _state_lock:
            _state.update({
                "enabled":        True,
                "batch_loras":    prescan,
                "image_index":    0,
                "history_weight": history_weight,
                "debug_log":      debug_log,
            })

        if limit > 0 or pinned:
            to_prewarm = _choose_prewarm(prescan, max(limit, len(pinned)),
                                         history_weight, pinned)
            logger.info(f"[SmartLoRACache] Pre-warming: {to_prewarm}")
            _prewarm_cache(to_prewarm)

    # ------------------------------------------------------------------
    # HOOK 2 — process_before_every_sampling
    # Fires after antiwildcards — reads truly final prompt
    # ------------------------------------------------------------------
    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        history_weight: float,
        history_window: int,
        debug_log: bool,
        **kwargs,
    ):
        with _state_lock:
            if not _state["enabled"]:
                return
            idx = _state["image_index"]

        all_prompts = getattr(p, "all_prompts", None)
        if isinstance(all_prompts, list) and len(all_prompts) > idx:
            final_prompt = all_prompts[idx]
        else:
            final_prompt = getattr(p, "prompt", "") or ""

        final_loras = _extract_loras(final_prompt)

        with _state_lock:
            batch_loras = _state["batch_loras"]
            if idx < len(batch_loras):
                prev = batch_loras[idx]
                if final_loras != prev:
                    injected = final_loras - prev
                    removed  = prev - final_loras
                    if injected:
                        logger.debug(f"[SmartLoRACache] img {idx}: AW injected {injected}")
                    if removed:
                        logger.debug(f"[SmartLoRACache] img {idx}: AW removed {removed}")
                    _state["batch_loras"][idx] = final_loras

        _history_push(final_loras)
        logger.debug(f"[SmartLoRACache] img {idx} final LoRAs: {final_loras}")

    # ------------------------------------------------------------------
    def before_process_batch(
        self, p, enabled, history_weight, history_window, debug_log, **kwargs
    ):
        batch_number = kwargs.get("batch_number", 0)
        batch_size   = getattr(p, "batch_size", 1) or 1
        with _state_lock:
            _state["image_index"] = batch_number * batch_size

    def postprocess_image(
        self, p, pp, enabled, history_weight, history_window, debug_log
    ):
        with _state_lock:
            if not _state["enabled"]:
                return
            max_idx = max(0, len(_state["batch_loras"]) - 1)
            _state["image_index"] = min(_state["image_index"] + 1, max_idx)

    def postprocess(
        self, p, processed, enabled, history_weight, history_window, debug_log
    ):
        with _state_lock:
            _state["enabled"] = False

        if enabled and debug_log:
            with _history_lock:
                n   = len(_lora_history)
                top = sorted(_history_freq.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"[SmartLoRACache] Batch done. History: {n} imgs. Top: {top}"
            )
