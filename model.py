import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel
from typing import Dict, Optional, Tuple

from config import Config


class DepthDecoder(nn.Module):
    """
    Small autoregressive transformer over the CODEBOOK axis (RQ-Transformer depth model).

    For one frame it consumes a length-k sequence in the depth dimension:
        [ projection(h) , emb(c0) , emb(c1) , ... , emb(c_{k-2}) ]
    and predicts c1..c_{k-1} at output positions 1..k-1 (position 0 = the backbone hidden
    state is a context prefix; its output is unused). Causality over the k positions is the
    plain LlamaModel causal mask, so position j+1 only attends to {h, c0..c_j} → predicts c_{j+1}.

    Built from a LlamaModel fed via inputs_embeds (never input_ids): we get tested RoPE /
    RMSNorm / SDPA for free. The unused token embedding table is kept tiny (vocab_size=8).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        depth_cfg = LlamaConfig(
            hidden_size             = cfg.depth_hidden_size,
            num_hidden_layers       = cfg.depth_num_layers,
            num_attention_heads     = cfg.depth_num_heads,
            num_key_value_heads     = cfg.depth_num_heads,   # no GQA — sequence is length k
            intermediate_size       = cfg.depth_intermediate_size,
            vocab_size              = 8,                       # unused (inputs_embeds path)
            max_position_embeddings = cfg.k_codebooks + 2,
            rms_norm_eps            = 1e-5,
        )
        self.model = LlamaModel(depth_cfg)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """inputs_embeds [*, S, D] → last_hidden_state [*, S, D]."""
        return self.model(inputs_embeds=inputs_embeds, use_cache=False).last_hidden_state


class TTSModel(nn.Module):
    """
    LLM backbone + RQ depth-transformer over Mimi codebooks (CSM/Marvis-style).

    The backbone (any HF causal LM) predicts codebook 0 (Mimi's semantic codebook) at each
    frame; a small DepthDecoder predicts codebooks 1..k-1 autoregressively within the frame,
    conditioned on the backbone hidden state + the previously-generated codebooks. There is
    NO delay pattern — the backbone runs at sequence length text + T (+1 EOS slot).

    Sequence layout (via inputs_embeds):
      [ text tokens... | <audio_start> | target audio frames | EOS slot ]
      (with optional [ <reference_start> | ref frames | <reference_end> ] prefix)

    Shapes:
      input_ids    [B, L]    text-vocab IDs at text positions, 0 at audio positions (unused)
      audio_codes  [B, L, k] per-codebook input code; PAD (value >= codebook_size) at text/edges
      audio_mask   [B, L]    True at audio frames (ref or target)
      labels       [B, L, k] per-codebook target; -100 at text/ref/edges; cb0 = AUDIO_EOS at EOS slot
    """

    def __init__(self, cfg: Config, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.k   = cfg.k_codebooks

        # Load LLM backbone in fp32; mixed precision is applied via autocast in the trainer.
        # (transformers >=5 loads in the checkpoint's native dtype by default — force fp32 so
        # the whole model, incl. the depth decoder and audio heads, shares one dtype.)
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name, dtype=torch.float32)

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
        cs     = cfg.codebook_size
        k      = self.k

        # AUDIO_PAD is an INPUT embedding index — the single extra row at the end of the
        # tied table, used for any PAD slot. Codes arrive in audio_codes with a PAD sentinel
        # value of `codebook_size` (what the collator/dataset write); the model detects
        # `code >= codebook_size` and remaps to this row.
        # AUDIO_EOS is an OUTPUT class on cb0's head ("stop generating"); it is never fed
        # back as an input.
        self.AUDIO_PAD = k * cs        # embedding-table index for PAD
        self.AUDIO_EOS = cs            # cb0 output class

        # One TIED offset embedding table: input index for code c in codebook q = c + q*cs.
        # Feeds both the backbone input (summed across codebooks) and the depth decoder.
        self.audio_embed = nn.Embedding(k * cs + 1, hidden)
        if not cfg.tie_audio_embeddings:
            self.depth_audio_embed = nn.Embedding(k * cs + 1, hidden)
        self.register_buffer("cb_offsets", torch.arange(k) * cs, persistent=False)  # [k]

        # Scale the summed input embedding by 1/sqrt(k) so its variance matches a single table.
        self.embed_scale = 1.0 / math.sqrt(self.k)

        # cb0 predictor on the backbone (extra class = AUDIO_EOS).
        self.codebook0_head = nn.Linear(hidden, cs + 1, bias=False)

        # Depth path: project the backbone hidden into the decoder width, then k-1 per-position
        # heads (audio_heads[j] predicts codebook j+1).
        self.projection  = nn.Linear(hidden, cfg.depth_hidden_size, bias=False)
        self.audio_heads = nn.ModuleList([
            nn.Linear(cfg.depth_hidden_size, cs, bias=False) for _ in range(k - 1)
        ])
        self.depth_decoder = DepthDecoder(cfg)

        # Initialize new parameters with the LLM's init std
        init_std = getattr(self.llm.config, "initializer_range", 0.02)
        nn.init.normal_(self.audio_embed.weight, mean=0.0, std=init_std)
        if not cfg.tie_audio_embeddings:
            nn.init.normal_(self.depth_audio_embed.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.codebook0_head.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)
        for head in self.audio_heads:
            nn.init.normal_(head.weight, mean=0.0, std=init_std)

        # Optional LoRA wrapping (backbone only; depth decoder is trained fully)
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
    # Embedding-table helpers (tie-aware)
    # ------------------------------------------------------------------

    def _input_table(self) -> nn.Embedding:
        return self.audio_embed

    def _depth_table(self) -> nn.Embedding:
        return self.audio_embed if self.cfg.tie_audio_embeddings else self.depth_audio_embed

    def _code_indices(self, codes: torch.LongTensor) -> torch.LongTensor:
        """Map raw per-codebook codes [..., k] to tied-table indices, PAD (value>=cs) → AUDIO_PAD."""
        is_pad = codes >= self.cfg.codebook_size
        offset = codes + self.cb_offsets
        return torch.where(is_pad, torch.full_like(codes, self.AUDIO_PAD), offset)

    # ------------------------------------------------------------------
    # Backbone forward helpers
    # ------------------------------------------------------------------

    def _build_inputs_embeds(
        self,
        input_ids:   torch.LongTensor,   # [B, L]
        audio_codes: torch.LongTensor,   # [B, L, k]
        audio_mask:  torch.BoolTensor,   # [B, L]
    ) -> torch.Tensor:
        """Build [B, L, H] inputs_embeds mixing text and summed-audio embeddings (no delay)."""
        text_vocab  = self.llm.config.vocab_size
        text_embeds = self.llm.model.embed_tokens(input_ids.clamp(0, text_vocab - 1))  # [B, L, H]

        idx       = self._code_indices(audio_codes)                 # [B, L, k]
        audio_sum = self._input_table()(idx).sum(dim=2) * self.embed_scale  # [B, L, H]
        audio_sum = audio_sum.to(text_embeds.dtype)

        return torch.where(audio_mask.unsqueeze(-1), audio_sum, text_embeds)

    def backbone_hidden(
        self,
        input_ids:      torch.LongTensor,
        audio_codes:    torch.LongTensor,
        audio_mask:     torch.BoolTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """Run the backbone over the full sequence → last_hidden_state [B, L, H]."""
        emb = self._build_inputs_embeds(input_ids, audio_codes, audio_mask)
        return self.llm.model(
            inputs_embeds=emb, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state

    def _depth_logits(self, h_sel: torch.Tensor, codes_sel: torch.LongTensor):
        """
        Teacher-forced depth pass for a batch of frames.

        Args:
            h_sel:     [M, H] backbone hidden state per frame (predicts that frame's codes)
            codes_sel: [M, k] real codes c0..c_{k-1} for those frames
        Returns:
            list of k-1 logits tensors, logits[j] = [M, cs] predicting c_{j+1}
        """
        M = h_sel.shape[0]
        tf_idx = codes_sel[:, : self.k - 1] + self.cb_offsets[: self.k - 1]  # [M, k-1] (c0..c_{k-2})
        tf_emb = self._depth_table()(tf_idx)                                 # [M, k-1, H]
        prefix = h_sel.unsqueeze(1)                                          # [M, 1, H]
        depth_in_H = torch.cat([prefix, tf_emb], dim=1)                      # [M, k, H]
        depth_in   = self.projection(depth_in_H)                            # [M, k, D]
        dec = self.depth_decoder(depth_in)                                  # [M, k, D]
        return [self.audio_heads[j](dec[:, j + 1, :].float()) for j in range(self.k - 1)]

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

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
            total_loss: cb0_loss_weight * cb0_CE + depth_loss_weight * depth_CE
            loss_dict:  {'loss', 'loss_cb0', 'loss_depth'}
        """
        hidden = self.backbone_hidden(input_ids, audio_codes, audio_mask, attention_mask)

        # Causal shift: hidden at position i predicts labels at position i+1
        pred_hidden = hidden[:, :-1, :]     # [B, L-1, H]
        target      = labels[:, 1:, :]      # [B, L-1, k]

        loss_dict: Dict[str, float] = {}

        # ---- cb0 loss (backbone; includes EOS) ----
        cb0_t  = target[:, :, 0]            # [B, L-1]
        valid0 = cb0_t != -100
        if valid0.any():
            h0      = pred_hidden[valid0]                       # [N0, H]
            logits0 = self.codebook0_head(h0.float())           # [N0, cs+1]
            t0      = cb0_t[valid0]                              # [N0]
            if self.cfg.eos_loss_weight != 1.0:
                w = torch.ones(self.cfg.codebook_size + 1, device=logits0.device)
                w[self.AUDIO_EOS] = self.cfg.eos_loss_weight
                cb0_loss = F.cross_entropy(logits0, t0, weight=w)
            else:
                cb0_loss = F.cross_entropy(logits0, t0)
        else:
            cb0_loss = pred_hidden.sum() * 0.0
        loss_dict["loss_cb0"] = float(cb0_loss.detach())

        # ---- depth loss (cb1..cb_{k-1}); supervised real frames only ----
        depth_mask = target[:, :, 1] != -100                    # [B, L-1] real frames
        if self.training and self.cfg.depth_train_fraction < 1.0:
            keep = torch.rand(depth_mask.shape, device=depth_mask.device) < self.cfg.depth_train_fraction
            depth_mask = depth_mask & keep
        if depth_mask.any() and self.k > 1:
            h_sel     = pred_hidden[depth_mask]                 # [M, H]
            codes_sel = target[depth_mask]                      # [M, k] (all real)
            logits_list = self._depth_logits(h_sel, codes_sel)  # list of [M, cs]
            depth_tgt   = codes_sel[:, 1:]                      # [M, k-1]
            terms = [F.cross_entropy(logits_list[j], depth_tgt[:, j]) for j in range(self.k - 1)]
            depth_loss = torch.stack(terms).mean()
        else:
            depth_loss = pred_hidden.sum() * 0.0
        loss_dict["loss_depth"] = float(depth_loss.detach())

        total_loss = self.cfg.cb0_loss_weight * cb0_loss + self.cfg.depth_loss_weight * depth_loss
        loss_dict["loss"] = float(total_loss.detach())
        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # DPO-readiness hooks (log-probs + freezing) — DPO itself not implemented
    # ------------------------------------------------------------------

    def compute_logprobs(
        self,
        input_ids:      torch.LongTensor,
        audio_codes:    torch.LongTensor,
        audio_mask:     torch.BoolTensor,
        labels:         torch.LongTensor,
        attention_mask: torch.LongTensor,
        include_depth:  bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Single differentiable teacher-forced pass returning per-frame log-probabilities.

        cb0 is the "content stream" (analogue of Moshi's text stream that Kyutai's GRPO
        targets). No no_grad / argmax / loss reduction — suitable for DPO/GRPO objectives.

        Returns:
            cb0_logprobs     [B, L-1]  log p(cb0 target); 0 at non-cb0-supervised positions
            cb0_mask         [B, L-1]  bool, True where cb0 is supervised
            cb0_logprob_sum  [B]       per-sequence sum of cb0 log-probs
            (if include_depth)
            full_logprobs    [B, L-1]  cb0 + sum_j log p(c_{j+1}) at real frames
            full_logprob_sum [B]
        """
        hidden      = self.backbone_hidden(input_ids, audio_codes, audio_mask, attention_mask)
        pred_hidden = hidden[:, :-1, :]     # [B, L-1, H]
        target      = labels[:, 1:, :]      # [B, L-1, k]
        B, Lm1, _   = target.shape

        logp0    = F.log_softmax(self.codebook0_head(pred_hidden.float()), dim=-1)  # [B,L-1,cs+1]
        cb0_mask = target[:, :, 0] != -100                                          # [B,L-1]
        tgt0     = target[:, :, 0].clamp_min(0)
        cb0_logprobs = logp0.gather(-1, tgt0.unsqueeze(-1)).squeeze(-1)             # [B,L-1]
        cb0_logprobs = cb0_logprobs.masked_fill(~cb0_mask, 0.0)

        out = {
            "cb0_logprobs":    cb0_logprobs,
            "cb0_mask":        cb0_mask,
            "cb0_logprob_sum": cb0_logprobs.sum(dim=1),
        }

        if include_depth and self.k > 1:
            depth_mask = target[:, :, 1] != -100                                    # [B,L-1]
            full = cb0_logprobs.clone()
            if depth_mask.any():
                h_sel     = pred_hidden[depth_mask]                                 # [M,H]
                codes_sel = target[depth_mask]                                      # [M,k]
                logits_list = self._depth_logits(h_sel, codes_sel)                  # list [M,cs]
                depth_tgt   = codes_sel[:, 1:]                                      # [M,k-1]
                depth_lp = torch.zeros(h_sel.shape[0], device=full.device)          # [M]
                for j in range(self.k - 1):
                    lp = F.log_softmax(logits_list[j], dim=-1)
                    depth_lp = depth_lp + lp.gather(-1, depth_tgt[:, j:j + 1]).squeeze(-1)
                # scatter [M] back into [B,L-1] at the masked positions
                full[depth_mask] = full[depth_mask] + depth_lp
            out["full_logprobs"]    = full
            out["full_logprob_sum"] = full.sum(dim=1)

        return out

    def freeze_depth_decoder(self):
        """Freeze the depth path (decoder + projection + audio_heads). Backbone, tied table,
        and codebook0_head stay trainable — mirrors CSM-style targeted RL on the content stream."""
        for p in self.depth_decoder.parameters():
            p.requires_grad_(False)
        for p in self.projection.parameters():
            p.requires_grad_(False)
        for head in self.audio_heads:
            for p in head.parameters():
                p.requires_grad_(False)

    def unfreeze_depth_decoder(self):
        for p in self.depth_decoder.parameters():
            p.requires_grad_(True)
        for p in self.projection.parameters():
            p.requires_grad_(True)
        for head in self.audio_heads:
            for p in head.parameters():
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _audio_embed_frame(self, frame: torch.LongTensor) -> torch.Tensor:
        """Backbone input embedding for one full frame [k] → [1, 1, H] (summed, scaled)."""
        idx = self._code_indices(frame)                      # [k]
        emb = self._input_table()(idx).sum(dim=0)            # [H]
        return (emb * self.embed_scale).view(1, 1, -1)

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
        Autoregressively generate Mimi codes for `text`.

        Returns:
            codes: LongTensor [k, n_frames] (may be [k, 0] on empty generation)
        """
        device = next(self.parameters()).device
        self.eval()

        text_vocab   = self.llm.config.vocab_size
        embed_tokens = self.llm.model.embed_tokens
        llm_dtype    = next(self.llm.parameters()).dtype

        prompt_embeds_list = []

        # --- Optional reference block: <reference_start> ref frames <reference_end> ---
        if ref_codes is not None and reference_start_id is not None and reference_end_id is not None:
            ref_codes = ref_codes.to(device)
            start_t = torch.tensor([[reference_start_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(start_t.clamp(0, text_vocab - 1)))
            for t in range(ref_codes.shape[1]):
                prompt_embeds_list.append(self._audio_embed_frame(ref_codes[:, t]))
            end_t = torch.tensor([[reference_end_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(end_t.clamp(0, text_vocab - 1)))

        # --- Text + <audio_start> ---
        text_ids  = tokenizer.encode(text, add_special_tokens=False)
        full_ids  = text_ids + [self.audio_start_id]
        id_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        prompt_embeds_list.append(embed_tokens(id_tensor.clamp(0, text_vocab - 1)))

        # Prefill the backbone
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1).to(llm_dtype)
        out     = self.llm.model(inputs_embeds=prompt_embeds, use_cache=True)
        hidden  = out.last_hidden_state
        past_kv = out.past_key_values

        frames: list = []
        for f in range(max_audio_frames):
            h = hidden[:, -1:, :]  # [1, 1, H] predicts frame f

            # --- cb0 from the backbone ---
            logits0 = self.codebook0_head(h.float()).squeeze(1)  # [1, cs+1]
            if f < min_audio_frames:
                logits0[:, self.AUDIO_EOS] = float("-inf")
            c0 = _sample(logits0, temperature, top_k, top_p)
            if c0.item() == self.AUDIO_EOS:
                break

            # --- cb1..cb_{k-1} from the depth decoder (re-run over the growing depth seq) ---
            frame = [int(c0.item())]
            proj_h = self.projection(h.to(llm_dtype))                    # [1, 1, D]
            depth_embs = [proj_h]
            for j in range(self.k - 1):
                c_j     = frame[j]
                tok_idx = torch.tensor([c_j + int(self.cb_offsets[j].item())],
                                       dtype=torch.long, device=device)
                tok_emb = self._depth_table()(tok_idx).view(1, 1, -1)     # [1, 1, H]
                depth_embs.append(self.projection(tok_emb.to(llm_dtype)))
                seq = torch.cat(depth_embs, dim=1)                        # [1, j+2, D]
                dec = self.depth_decoder(seq)[:, -1, :]                   # [1, D]
                logits_j = self.audio_heads[j](dec.float())              # [1, cs]
                c_next   = _sample(logits_j, temperature, top_k, top_p)
                frame.append(int(c_next.item()))

            frame_t = torch.tensor(frame, dtype=torch.long, device=device)  # [k]
            frames.append(frame_t)

            # --- advance the backbone with this frame's summed embedding ---
            nxt = self._audio_embed_frame(frame_t).to(llm_dtype)
            out     = self.llm.model(inputs_embeds=nxt, past_key_values=past_kv, use_cache=True)
            hidden  = out.last_hidden_state
            past_kv = out.past_key_values

        if not frames:
            return torch.zeros(self.k, 0, dtype=torch.long)
        return torch.stack(frames, dim=1).to("cpu", torch.long)  # [k, n_frames]


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
