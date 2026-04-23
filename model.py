import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Dict, Optional, Tuple

from config import Config


class TTSModel(nn.Module):
    """
    LLM fine-tuned to predict Mimi codebook tokens in MusicGen-style DELAY pattern.

    At each audio step s the model sees the sum of k per-codebook input embeddings
    (one per codebook for the codes placed at step s by the delay transform) and
    predicts k tokens via k parallel heads. Codebook q at real frame f lives at
    step s = f + q.

    Sequence layout (via inputs_embeds):
      [ text tokens... | <audio_start> | delayed target audio steps ]
      (with optional [ <reference_start> | delayed ref | <reference_end> ] prefix)

    Shapes:
      input_ids    [B, L]    text-vocab IDs at text positions, 0 at audio positions (unused)
      audio_codes  [B, L, k] per-codebook input code; AUDIO_PAD at text and invalid delay edges
      audio_mask   [B, L]    True at audio steps
      labels       [B, L, k] per-codebook target; -100 at text/ref/invalid; AUDIO_EOS for cb0 at step T
    """

    def __init__(self, cfg: Config, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.k   = cfg.k_codebooks

        # Load LLM backbone in fp32; mixed precision is applied via autocast in the trainer.
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)

        # Resize embeddings to include the new special tokens
        new_vocab_size = len(tokenizer)
        self.llm.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8)

        # Special-token IDs (text-vocab). New-name set with old-name fallback so both
        # fresh tokenizers and legacy ones load without custom plumbing.
        def _lookup(*names):
            for name in names:
                tid = tokenizer.convert_tokens_to_ids(name)
                if tid is not None and tid != tokenizer.unk_token_id:
                    return tid
            return None
        self.audio_start_id     = _lookup("<|audio_start|>", "<|audio_sep|>")
        self.reference_start_id = _lookup("<|reference_start|>", "<|audio_start|>")
        self.reference_end_id   = _lookup("<|reference_end|>", "<|audio_end|>")

        hidden = self.llm.config.hidden_size

        # AUDIO_PAD is an INPUT index (embedding lookup for invalid delay positions).
        # AUDIO_EOS is an OUTPUT class in cb0's head ("stop generating").
        # They share the numeric value codebook_size (=2048) but live in different spaces
        # and never collide (EOS is never fed back as input after cb0 predicts it).
        self.AUDIO_PAD = cfg.codebook_size
        self.AUDIO_EOS = cfg.codebook_size

        # k input embedding tables, size codebook_size + 1 (extra row = PAD at index codebook_size).
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(cfg.codebook_size + 1, hidden)
            for _ in range(cfg.k_codebooks)
        ])

        # k prediction heads. cb0 has codebook_size + 1 outputs (extra = AUDIO_EOS);
        # cb1..cb_{k-1} have codebook_size outputs (no PAD/EOS in output space — PAD positions
        # get -100 labels and are skipped in the CE loss).
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden, cfg.codebook_size + (1 if i == 0 else 0), bias=False)
            for i in range(cfg.k_codebooks)
        ])

        # Scale the summed input embedding by 1/sqrt(k) so its variance matches
        # a single embedding table (each table init'd with std=init_std).
        self.embed_scale = 1.0 / math.sqrt(self.k)

        # Initialize new parameters with the LLM's init std
        init_std = getattr(self.llm.config, "initializer_range", 0.02)
        for emb in self.audio_embeds:
            nn.init.normal_(emb.weight, mean=0.0, std=init_std)
        for head in self.audio_heads:
            nn.init.normal_(head.weight, mean=0.0, std=init_std)

        # Optional LoRA wrapping
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules,
                lora_dropout=0.05,
                bias="none",
            )
            self.llm = get_peft_model(self.llm, lora_cfg)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def _build_inputs_embeds(
        self,
        input_ids:   torch.LongTensor,   # [B, L]
        audio_codes: torch.LongTensor,   # [B, L, k]
        audio_mask:  torch.BoolTensor,   # [B, L]
    ) -> torch.Tensor:
        """Build [B, L, H] inputs_embeds mixing text and summed-audio embeddings."""
        text_vocab = self.llm.config.vocab_size
        safe_ids   = input_ids.clamp(0, text_vocab - 1)
        text_embeds = self.llm.model.embed_tokens(safe_ids)  # [B, L, H]

        # Sum per-codebook embeddings at every position (even text positions — they hold
        # PAD at index codebook_size, which is a valid embedding row). We mask in the
        # text embeddings at non-audio positions via torch.where.
        audio_sum = self.audio_embeds[0](audio_codes[:, :, 0])
        for q in range(1, self.k):
            audio_sum = audio_sum + self.audio_embeds[q](audio_codes[:, :, q])
        audio_sum = audio_sum * self.embed_scale
        audio_sum = audio_sum.to(text_embeds.dtype)

        # Mix by audio_mask
        return torch.where(audio_mask.unsqueeze(-1), audio_sum, text_embeds)

    def forward(
        self,
        input_ids:      torch.LongTensor,      # [B, L]
        audio_codes:    torch.LongTensor,      # [B, L, k]
        audio_mask:     torch.BoolTensor,      # [B, L]
        labels:         torch.LongTensor,      # [B, L, k]
        attention_mask: torch.LongTensor,      # [B, L]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: weighted mean CE loss across codebooks
            loss_dict:  {'loss': ..., 'loss_cb0': ..., 'loss_cb1': ..., ...}
        """
        inputs_embeds = self._build_inputs_embeds(input_ids, audio_codes, audio_mask)

        out = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state  # [B, L, H]

        # Causal LM shift: hidden at position i predicts labels at position i+1
        pred_hidden   = hidden[:, :-1, :]       # [B, L-1, H]
        target_labels = labels[:, 1:, :]        # [B, L-1, k]

        # Resolve per-codebook weights to length k: truncate, or pad with the last value.
        raw_w = list(self.cfg.cb_loss_weights) if self.cfg.cb_loss_weights else [1.0]
        cb_w  = raw_w[: self.k] + [raw_w[-1]] * max(0, self.k - len(raw_w))

        weighted_terms: list = []
        total_weight: float  = 0.0
        loss_dict: Dict[str, float] = {}

        for cb_idx in range(self.k):
            target_k = target_labels[:, :, cb_idx]          # [B, L-1]
            valid    = target_k != -100                      # bool [B, L-1]
            if not valid.any():
                continue
            h_k      = pred_hidden[valid]                    # [N, H]
            logits_k = self.audio_heads[cb_idx](h_k.float())  # [N, out_dim]
            t_k      = target_k[valid]                       # [N]
            if cb_idx == 0 and self.cfg.eos_loss_weight != 1.0:
                # Upweight AUDIO_EOS to counter the 1:~(T) imbalance in cb0 supervision.
                eos_w = torch.ones(self.cfg.codebook_size + 1, device=logits_k.device)
                eos_w[self.AUDIO_EOS] = self.cfg.eos_loss_weight
                loss_k = F.cross_entropy(logits_k, t_k, weight=eos_w)
            else:
                loss_k = F.cross_entropy(logits_k, t_k)
            loss_dict[f"loss_cb{cb_idx}"] = loss_k.item()
            w = float(cb_w[cb_idx])
            weighted_terms.append(w * loss_k)
            total_weight += w

        if not weighted_terms:
            total_loss = torch.tensor(0.0, device=hidden.device, requires_grad=True)
        else:
            total_loss = torch.stack(weighted_terms).sum() / max(total_weight, 1e-8)

        loss_dict["loss"] = total_loss.item()
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Generation (delay-aware)
    # ------------------------------------------------------------------

    def _audio_embed_step(self, codes_step: torch.LongTensor) -> torch.Tensor:
        """
        Build the input embedding for a single audio step.

        Args:
            codes_step: [1, k] codes (one per codebook) for this step.
        Returns:
            [1, 1, H] input embedding, scaled.
        """
        acc = self.audio_embeds[0](codes_step[:, 0:1])            # [1, 1, H]
        for q in range(1, self.k):
            acc = acc + self.audio_embeds[q](codes_step[:, q:q + 1])
        return acc * self.embed_scale

    @torch.no_grad()
    def generate(
        self,
        text: str,
        tokenizer,
        max_audio_frames:    int   = 200,
        min_audio_frames:    int   = 10,
        temperature:         float = 1.0,
        top_k:               int   = 50,
        top_p:               float = 0.9,
        ref_codes:           Optional[torch.LongTensor] = None,   # [k, T_ref]
        reference_start_id:  Optional[int] = None,
        reference_end_id:    Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Autoregressively generate Mimi codes for the given text using the delay pattern.

        Returns:
            codes: LongTensor [k, n_frames]
        """
        device = next(self.parameters()).device
        self.eval()

        text_vocab   = self.llm.config.vocab_size
        embed_tokens = self.llm.model.embed_tokens
        llm_dtype    = next(self.llm.parameters()).dtype

        prompt_embeds_list = []

        # --- Optional reference block: <reference_start> ref_delayed <reference_end> ---
        if ref_codes is not None and reference_start_id is not None and reference_end_id is not None:
            ref_codes = ref_codes.to(device)
            T_ref     = ref_codes.shape[1]

            start_t = torch.tensor([[reference_start_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(start_t.clamp(0, text_vocab - 1)))  # [1, 1, H]

            # Apply delay to ref: [k, T_ref] -> [k, T_ref + k - 1]
            from dataset import apply_delay  # avoid import cycle at module load
            ref_delayed = apply_delay(ref_codes.cpu(), self.k, self.AUDIO_PAD).to(device)  # [k, L_ref]
            L_ref = ref_delayed.shape[1]
            # Embed each step (summed across codebooks, scaled)
            for s in range(L_ref):
                codes_step = ref_delayed[:, s:s + 1].T.contiguous()  # [1, k]
                prompt_embeds_list.append(self._audio_embed_step(codes_step))

            end_t = torch.tensor([[reference_end_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(end_t.clamp(0, text_vocab - 1)))

        # --- Text + <audio_start> ---
        text_ids  = tokenizer.encode(text, add_special_tokens=False)
        full_ids  = text_ids + [self.audio_start_id]
        id_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        prompt_embeds_list.append(embed_tokens(id_tensor.clamp(0, text_vocab - 1)))  # [1, L_text, H]

        # Prefill the LLM
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1).to(llm_dtype)
        out     = self.llm.model(inputs_embeds=prompt_embeds, use_cache=True)
        hidden  = out.last_hidden_state
        past_kv = out.past_key_values

        # --- Delay-pattern autoregressive loop ---
        # outputs[q] accumulates emitted codes for codebook q (one per step, including PAD).
        # After generation, un-delay to recover the [k, T] frame matrix.
        outputs: list = [[] for _ in range(self.k)]
        eos_step: Optional[int] = None
        max_steps = max_audio_frames + self.k - 1

        for step in range(max_steps):
            # Compute all k heads from the last hidden state (parallel prediction).
            h = hidden[:, -1:, :]  # [1, 1, H]
            logits_per_cb = [self.audio_heads[q](h.float()).squeeze(1) for q in range(self.k)]  # each [1, out_q]

            # Suppress EOS until min_audio_frames have been emitted by cb0
            if step < min_audio_frames:
                logits_per_cb[0][:, self.AUDIO_EOS] = float("-inf")

            # Determine the next code per codebook
            next_codes = torch.empty(self.k, dtype=torch.long, device=device)
            for q in range(self.k):
                # Leading edge: cb_q's first real frame is at step q
                if step < q:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                # If cb0 has already emitted EOS, stop supervising cb0
                if q == 0 and eos_step is not None:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                # Trailing edge: cb_q's frame index (step - q) is past EOS?
                if q > 0 and eos_step is not None and (step - q) >= eos_step:
                    next_codes[q] = self.AUDIO_PAD
                    continue
                # Sample
                sampled = _sample(logits_per_cb[q], temperature, top_k, top_p)  # scalar LongTensor
                if q == 0 and sampled.item() == self.AUDIO_EOS:
                    eos_step = step
                    next_codes[q] = self.AUDIO_PAD
                else:
                    next_codes[q] = sampled

            for q in range(self.k):
                outputs[q].append(next_codes[q].item())

            # Stop once we've flushed all tail codebooks
            if eos_step is not None and step >= eos_step + self.k - 1:
                break

            # Build next input embedding and advance
            codes_step = next_codes.unsqueeze(0)  # [1, k]
            next_embed = self._audio_embed_step(codes_step).to(hidden.dtype)
            out     = self.llm.model(inputs_embeds=next_embed, past_key_values=past_kv, use_cache=True)
            hidden  = out.last_hidden_state
            past_kv = out.past_key_values

        # --- Un-delay ---
        # Number of real frames T is eos_step if set, else max_audio_frames.
        T = eos_step if eos_step is not None else max_audio_frames
        # Need at least step >= f + q for frame f's cb_q. Clip T so all needed entries exist.
        max_available = min(len(outputs[q]) - q for q in range(self.k))
        T = min(T, max_available)
        if T <= 0:
            return torch.zeros(self.k, 0, dtype=torch.long)
        codes = torch.empty(self.k, T, dtype=torch.long)
        for q in range(self.k):
            codes[q] = torch.tensor(outputs[q][q:q + T], dtype=torch.long)
        return codes


def _sample(
    logits:      torch.Tensor,   # [1, vocab]
    temperature: float = 1.0,
    top_k:       int   = 50,
    top_p:       float = 0.9,
) -> torch.LongTensor:
    """Temperature + top-k + top-p sampling. Returns scalar LongTensor."""
    if temperature <= 0:
        return logits.argmax(-1).view(-1)
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k_ = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k_)
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)  # [1]
