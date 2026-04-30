"""
Wren-TTS model — a transformers-compatible wrapper over SmolLM2 + Mimi codebook heads.

Designed for use with `AutoModel.from_pretrained(..., trust_remote_code=True)`.
Self-contained: no imports from a `src/` folder.

Sequence layout (MusicGen-style delay pattern):

  [ text tokens... | <audio_start> | tgt_delay_steps ]
  + optional [ <reference_start> | ref_delay_steps | <reference_end> ] prefix

At each audio step s the model sees the sum of k per-codebook input embeddings
(scaled by 1/sqrt(k)) and predicts k tokens via k parallel heads. Codebook q at
frame f lives at step s = f + q. AUDIO_EOS (=codebook_size) is cb0's stop class;
cb0 emits it at step T (one past the last real frame).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

try:
    from .configuration_wren import WrenConfig          # package context (HF trust_remote_code)
except ImportError:
    # Flat-file context (push.py adds hf/ to sys.path). Use importlib so transformers'
    # `check_imports` static scanner doesn't misread this as an external pip dependency.
    import importlib
    WrenConfig = importlib.import_module("configuration_wren").WrenConfig


class WrenForTTS(PreTrainedModel):
    config_class      = WrenConfig
    base_model_prefix = "wren"

    def __init__(self, config: WrenConfig):
        super().__init__(config)
        self.k          = config.k_codebooks
        self.AUDIO_EOS  = config.codebook_size
        self.AUDIO_PAD  = config.codebook_size

        # Build backbone from its config only. Pretrained weights for the backbone
        # are included in our own state_dict, so no need to re-download here.
        llm_cfg            = AutoConfig.from_pretrained(config.llm_name)
        llm_cfg.vocab_size = config.vocab_size
        self.llm           = AutoModelForCausalLM.from_config(llm_cfg)

        hidden = self.llm.config.hidden_size

        # Input tables: codebook_size + 1 (extra row at index codebook_size = AUDIO_PAD).
        # Output heads: cb0 has codebook_size + 1 (extra class = AUDIO_EOS); cb1..cb_{k-1}
        # have codebook_size (no PAD/EOS output — PAD positions have -100 labels in training).
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(config.codebook_size + 1, hidden)
            for _ in range(self.k)
        ])
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden, config.codebook_size + (1 if i == 0 else 0), bias=False)
            for i in range(self.k)
        ])

        self.embed_scale = 1.0 / math.sqrt(self.k)
        self._mimi = None  # lazy-loaded on first use

    # --- Mimi codec (lazy-loaded, decoder + encoder used for audio I/O) ---

    @property
    def mimi(self):
        if self._mimi is None:
            from transformers import MimiModel
            self._mimi = MimiModel.from_pretrained(self.config.mimi_model_name).to(self.device)
            self._mimi.eval()
            for p in self._mimi.parameters():
                p.requires_grad_(False)
        return self._mimi

    @torch.no_grad()
    def encode_audio(
        self,
        waveform:         torch.Tensor,
        src_sample_rate:  int = 24000,
    ) -> torch.LongTensor:
        """Encode a reference waveform to Mimi codes (for voice cloning)."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if src_sample_rate != self.config.sampling_rate:
            import torchaudio.transforms as T
            waveform = T.Resample(src_sample_rate, self.config.sampling_rate)(waveform)
        x   = waveform.unsqueeze(0).to(self.device)
        out = self.mimi.encode(x, num_quantizers=self.k)
        return out.audio_codes[0].cpu()  # [k, n_frames]

    @torch.no_grad()
    def decode_audio(self, codes: torch.LongTensor) -> torch.Tensor:
        """Decode Mimi codes to 24 kHz waveform [1, T]."""
        if codes.numel() == 0:
            return torch.zeros(1, 0)
        codes_b = codes.unsqueeze(0).to(self.device)
        out     = self.mimi.decode(codes_b)
        return out.audio_values[0].cpu()

    # --- Generation ---

    def _audio_embed_step(self, codes_step: torch.LongTensor) -> torch.Tensor:
        """Summed per-codebook embedding for one audio step. codes_step: [1, k]. Returns [1, 1, H]."""
        acc = self.audio_embeds[0](codes_step[:, 0:1])
        for q in range(1, self.k):
            acc = acc + self.audio_embeds[q](codes_step[:, q:q + 1])
        return acc * self.embed_scale

    @torch.no_grad()
    def generate(
        self,
        input_ids:         torch.LongTensor,
        ref_codes:         Optional[torch.LongTensor] = None,
        max_audio_frames:  int   = 200,
        min_audio_frames:  int   = 2,
        temperature:       float = 0.8,
        top_k:             int   = 50,
        top_p:             float = 0.9,
        eos_bias:          float = 0.0,
        output_audio:      bool  = False,
        **kwargs,
    ):
        """
        Generate Mimi codes (or waveform) from a tokenized prompt.

        Args:
            input_ids:        [1, L] — text tokens ending with <|audio_start|>, as produced by WrenProcessor.
            ref_codes:        optional [k, T_ref] reference codes for voice cloning.
            max_audio_frames: hard cap on output length.
            min_audio_frames: suppress EOS for this many frames.
            eos_bias:         additive bias on cb0's AUDIO_EOS logit; raise (2–6) to reduce tail hallucination.
            output_audio:     if True, return [1, T] waveform; else return [k, n_frames] codes.
        """
        device = next(self.parameters()).device
        self.eval()

        embed_tokens = self.llm.get_input_embeddings()
        text_vocab   = self.llm.config.vocab_size
        llm_dtype    = next(self.llm.parameters()).dtype

        prompt_embeds_list = []

        # Optional reference block: <reference_start> ref_delayed <reference_end>
        if ref_codes is not None:
            if self.config.reference_start_id is None or self.config.reference_end_id is None:
                raise ValueError("reference_start_id/reference_end_id missing from config; cannot use ref_codes")
            ref_codes = ref_codes.to(device)

            start_t = torch.tensor([[self.config.reference_start_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(start_t.clamp(0, text_vocab - 1)))

            T_ref = ref_codes.shape[1]
            L_ref = T_ref + self.k - 1
            ref_delayed = torch.full((self.k, L_ref), self.AUDIO_PAD, dtype=torch.long, device=device)
            for q in range(self.k):
                ref_delayed[q, q:q + T_ref] = ref_codes[q]

            for s in range(L_ref):
                codes_step = ref_delayed[:, s:s + 1].T.contiguous()  # [1, k]
                prompt_embeds_list.append(self._audio_embed_step(codes_step))

            end_t = torch.tensor([[self.config.reference_end_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(end_t.clamp(0, text_vocab - 1)))

        # Text prompt — already terminated by <|audio_start|> via the processor
        ids = input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        prompt_embeds_list.append(embed_tokens(ids.clamp(0, text_vocab - 1)))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1).to(llm_dtype)
        out     = self.llm.model(inputs_embeds=prompt_embeds, use_cache=True)
        hidden  = out.last_hidden_state
        past_kv = out.past_key_values

        # Delay-pattern autoregressive loop
        outputs: list = [[] for _ in range(self.k)]
        eos_step: Optional[int] = None
        max_steps = max_audio_frames + self.k - 1

        for step in range(max_steps):
            h = hidden[:, -1:, :]
            logits_per_cb = [self.audio_heads[q](h.float()).squeeze(1) for q in range(self.k)]

            logits_per_cb[0][:, self.AUDIO_EOS] = logits_per_cb[0][:, self.AUDIO_EOS] + eos_bias
            if step < min_audio_frames:
                logits_per_cb[0][:, self.AUDIO_EOS] = float("-inf")

            next_codes = torch.empty(self.k, dtype=torch.long, device=device)
            for q in range(self.k):
                if step < q:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                if q == 0 and eos_step is not None:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                if q > 0 and eos_step is not None and (step - q) >= eos_step:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                sampled = _sample(logits_per_cb[q], temperature, top_k, top_p)
                if q == 0 and sampled.item() == self.AUDIO_EOS:
                    eos_step = step
                    next_codes[q] = self.AUDIO_PAD
                else:
                    next_codes[q] = sampled

            for q in range(self.k):
                outputs[q].append(next_codes[q].item())

            if eos_step is not None and step >= eos_step + self.k - 1:
                break

            next_embed = self._audio_embed_step(next_codes.unsqueeze(0)).to(hidden.dtype)
            out     = self.llm.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
            hidden  = out.last_hidden_state
            past_kv = out.past_key_values

        # Un-delay into [k, T]
        T = eos_step if eos_step is not None else max_audio_frames
        max_available = min(len(outputs[q]) - q for q in range(self.k))
        T = min(T, max_available)
        if T <= 0:
            codes = torch.zeros(self.k, 0, dtype=torch.long)
        else:
            codes = torch.empty(self.k, T, dtype=torch.long)
            for q in range(self.k):
                codes[q] = torch.tensor(outputs[q][q:q + T], dtype=torch.long)

        if output_audio:
            return self.decode_audio(codes)
        return codes


def _sample(
    logits:      torch.Tensor,
    temperature: float = 1.0,
    top_k:       int   = 50,
    top_p:       float = 0.9,
) -> torch.LongTensor:
    """Temperature + top-k + top-p sampling. Returns a 1-D LongTensor."""
    if temperature <= 0:
        return logits.argmax(-1).view(-1)
    logits = logits / temperature

    if top_k and top_k > 0:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
        logits = logits.masked_fill(logits < v[..., -1:], float("-inf"))

    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum   = probs.cumsum(-1)
        mask  = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0]  = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).view(-1)
