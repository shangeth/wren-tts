"""
Wren-TTS model — a transformers-compatible wrapper over SmolLM2 + Mimi codebook heads.

Designed for use with `AutoModel.from_pretrained(..., trust_remote_code=True)`.
Self-contained: no imports from a `src/` folder.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

try:
    from .configuration_wren import WrenConfig          # package context (HF trust_remote_code)
except ImportError:
    from configuration_wren import WrenConfig           # flat context (our push script)


class WrenForTTS(PreTrainedModel):
    """
    SmolLM2 backbone + k separate audio-code embedding tables + k audio heads.

    Sequence layout fed to the LLM (via inputs_embeds):
      [ text tokens... | <audio_sep> | cb0_f0 | cb1_f0 | ... | cbk_f0 | cb0_f1 | ... ]

    cb0's head has an extra output class `AUDIO_EOS = codebook_size` used as stop token.
    """

    config_class      = WrenConfig
    base_model_prefix = "wren"

    def __init__(self, config: WrenConfig):
        super().__init__(config)
        self.k          = config.k_codebooks
        self.AUDIO_EOS  = config.codebook_size

        # Build backbone from its config only. Pretrained weights for the backbone
        # are included in our own state_dict, so no need to re-download here.
        # Set vocab_size directly on the sub-config to avoid a subsequent
        # resize_token_embeddings call (which breaks under meta-tensor init).
        llm_cfg            = AutoConfig.from_pretrained(config.llm_name)
        llm_cfg.vocab_size = config.vocab_size
        self.llm           = AutoModelForCausalLM.from_config(llm_cfg)

        hidden = self.llm.config.hidden_size
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(config.codebook_size, hidden)
            for _ in range(self.k)
        ])
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden, config.codebook_size + (1 if i == 0 else 0), bias=False)
            for i in range(self.k)
        ])

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
            input_ids:        [1, L] — text tokens ending with <|audio_sep|>, as produced by WrenProcessor.
            ref_codes:        optional [k, T_ref] reference codes for voice cloning.
            max_audio_frames: hard cap on output length.
            min_audio_frames: suppress EOS for this many frames (prevents trivially-short outputs).
            eos_bias:         additive bias on AUDIO_EOS logit; raise (e.g. 2–6) to reduce hallucinated continuations.
            output_audio:     if True, return [1, T] waveform; else return [k, n_frames] codes.
        """
        device = next(self.parameters()).device
        self.eval()

        embed_tokens = self.llm.get_input_embeddings()
        text_vocab   = self.llm.config.vocab_size
        llm_dtype    = next(self.llm.parameters()).dtype

        prompt_embeds_list = []

        # Optional reference-audio block (voice cloning)
        if ref_codes is not None:
            if self.config.audio_start_id is None or self.config.audio_end_id is None:
                raise ValueError("audio_start_id/audio_end_id missing from config; cannot use ref_codes")
            ref_codes = ref_codes.to(device)
            start_t = torch.tensor([[self.config.audio_start_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(start_t.clamp(0, text_vocab - 1)))

            ref_tokens = ref_codes.T.reshape(-1)  # interleaved [T_ref * k]
            for pos, code in enumerate(ref_tokens):
                cb_idx = pos % self.k
                idx    = code.clamp(0, self.config.codebook_size - 1).unsqueeze(0)
                prompt_embeds_list.append(self.audio_embeds[cb_idx](idx).unsqueeze(0))

            end_t = torch.tensor([[self.config.audio_end_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(end_t.clamp(0, text_vocab - 1)))

        # Text prompt — already terminated by <|audio_sep|> via the processor
        ids = input_ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        prompt_embeds_list.append(embed_tokens(ids.clamp(0, text_vocab - 1)))

        prompt_embeds = torch.cat(prompt_embeds_list, dim=1).to(llm_dtype)
        out      = self.llm.model(inputs_embeds=prompt_embeds, use_cache=True)
        hidden   = out.last_hidden_state
        past_kv  = out.past_key_values

        all_codes, audio_pos, n_frames = [], 0, 0
        while n_frames < max_audio_frames:
            cb_idx = audio_pos % self.k
            h      = hidden[:, -1:, :]
            logits = self.audio_heads[cb_idx](h.float()).squeeze(1)

            if cb_idx == 0:
                logits[:, self.AUDIO_EOS] = logits[:, self.AUDIO_EOS] + eos_bias
                if n_frames < min_audio_frames:
                    logits[:, self.AUDIO_EOS] = float("-inf")

            next_code = _sample(logits, temperature, top_k, top_p)

            if cb_idx == 0 and next_code.item() == self.AUDIO_EOS:
                break

            all_codes.append(next_code.item())
            audio_pos += 1
            if audio_pos % self.k == 0:
                n_frames += 1

            embed_idx = next_code.clamp(0, self.config.codebook_size - 1)
            emb = self.audio_embeds[cb_idx](embed_idx.unsqueeze(0)).to(hidden.dtype)
            out     = self.llm.model(inputs_embeds=emb, past_key_values=past_kv, use_cache=True)
            hidden  = out.last_hidden_state
            past_kv = out.past_key_values

        complete = (len(all_codes) // self.k) * self.k
        if complete == 0:
            codes = torch.zeros(self.k, 0, dtype=torch.long)
        else:
            codes = torch.tensor(all_codes[:complete], dtype=torch.long)
            codes = codes.reshape(complete // self.k, self.k).T  # [k, n_frames]

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
