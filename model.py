import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Dict, Tuple

from config import Config


class TTSModel(nn.Module):
    """
    LLM fine-tuned to autoregressively predict interleaved Mimi codebook tokens.

    Architecture:
      - Pretrained LLM backbone (SmolLM2-360M by default)
      - k separate audio embedding tables: Embedding(codebook_size, hidden)
      - k separate prediction heads:        Linear(hidden, codebook_size)

    Sequence format fed to the LLM (via inputs_embeds):
      [ text tokens... | <audio_sep> | cb0_f0 | cb1_f0 | ... | cbk_f0 | cb0_f1 | ... ]

    At each audio position i (0-indexed within audio), codebook index = i % k.
    Text and sep positions use the LLM's own embed_tokens table.
    Audio positions use audio_embeds[cb_idx].

    Loss: mean cross-entropy across codebooks (causal LM shift applied).
    """

    def __init__(self, cfg: Config, tokenizer):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.k_codebooks

        # Load LLM backbone in fp32; mixed precision is applied via autocast in the trainer.
        self.llm = AutoModelForCausalLM.from_pretrained(cfg.llm_name)

        # Resize embeddings to include the new <|audio_sep|> token
        new_vocab_size = len(tokenizer)
        self.llm.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8)
        self.audio_sep_id = tokenizer.convert_tokens_to_ids("<|audio_sep|>")

        hidden = self.llm.config.hidden_size

        # k audio embedding tables (one per codebook)
        self.audio_embeds = nn.ModuleList([
            nn.Embedding(cfg.codebook_size, hidden)
            for _ in range(cfg.k_codebooks)
        ])

        # k audio prediction heads (one per codebook).
        # cb0 gets codebook_size+1 outputs: indices 0..2047 are codes, 2048 is AUDIO_EOS.
        # cb1..cbk-1 stay at codebook_size (EOS is only signalled via cb0).
        self.AUDIO_EOS = cfg.codebook_size  # = 2048
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden, cfg.codebook_size + (1 if i == 0 else 0), bias=False)
            for i in range(cfg.k_codebooks)
        ])

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

    def _build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,   # [B, L]
        audio_mask: torch.BoolTensor,  # [B, L]
        cb_indices: torch.LongTensor,  # [B, L]
    ) -> torch.Tensor:
        """Build [B, L, H] inputs_embeds mixing text and audio embeddings."""
        text_vocab = self.llm.config.vocab_size
        safe_ids = input_ids.clamp(0, text_vocab - 1)
        inputs_embeds = self.llm.model.embed_tokens(safe_ids).clone()  # [B, L, H]

        # Overwrite audio positions with per-codebook embeddings
        for cb_idx in range(self.k):
            mask_k = audio_mask & (cb_indices == cb_idx)  # [B, L]
            if not mask_k.any():
                continue
            code_vals = input_ids[mask_k].clamp(0, self.cfg.codebook_size - 1)
            emb_k = self.audio_embeds[cb_idx](code_vals)          # [N, H]
            inputs_embeds[mask_k] = emb_k.to(inputs_embeds.dtype)

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor,      # [B, L]
        audio_mask: torch.BoolTensor,     # [B, L]
        cb_indices: torch.LongTensor,     # [B, L]
        labels: torch.LongTensor,         # [B, L] — -100 at text/sep positions
        attention_mask: torch.LongTensor, # [B, L]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: mean CE loss across codebooks
            loss_dict:  {'loss': ..., 'loss_cb0': ..., 'loss_cb1': ..., ...}
        """
        inputs_embeds = self._build_inputs_embeds(input_ids, audio_mask, cb_indices)

        out = self.llm.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state  # [B, L, H]

        # Causal LM shift: hidden at position i predicts labels at position i+1
        pred_hidden     = hidden[:, :-1, :]       # [B, L-1, H]
        target_labels   = labels[:, 1:]           # [B, L-1]
        pred_audio_mask = audio_mask[:, 1:]       # [B, L-1]
        pred_cb_indices = cb_indices[:, 1:]       # [B, L-1]

        # Resolve per-codebook weights to length k: truncate, or pad with the last value.
        raw_w = list(self.cfg.cb_loss_weights) if self.cfg.cb_loss_weights else [1.0]
        cb_w  = raw_w[: self.k] + [raw_w[-1]] * max(0, self.k - len(raw_w))

        weighted_terms: list = []
        total_weight:   float = 0.0
        loss_dict: Dict[str, float] = {}

        for cb_idx in range(self.k):
            mask_k = pred_audio_mask & (pred_cb_indices == cb_idx)  # [B, L-1]
            if not mask_k.any():
                continue
            h_k      = pred_hidden[mask_k]                      # [N, H]
            logits_k = self.audio_heads[cb_idx](h_k.float())    # [N, codebook_size]
            target_k = target_labels[mask_k]                    # [N]
            valid    = target_k >= 0
            if valid.any():
                if cb_idx == 0 and self.cfg.eos_loss_weight != 1.0:
                    # Upweight the AUDIO_EOS class to counter the 1:~200 imbalance
                    # (one EOS target per ~100–300 normal audio-code targets in cb0).
                    eos_w = torch.ones(self.cfg.codebook_size + 1, device=logits_k.device)
                    eos_w[self.AUDIO_EOS] = self.cfg.eos_loss_weight
                    loss_k = F.cross_entropy(logits_k[valid], target_k[valid], weight=eos_w)
                else:
                    loss_k = F.cross_entropy(logits_k[valid], target_k[valid])
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

    @torch.no_grad()
    def generate(
        self,
        text: str,
        tokenizer,
        max_audio_frames: int = 200,
        min_audio_frames: int = 10,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        ref_codes: "torch.LongTensor | None" = None,  # [k, T_ref] reference audio
        audio_start_id: "int | None" = None,
        audio_end_id: "int | None" = None,
    ) -> torch.LongTensor:
        """
        Autoregressively generate Mimi codes for the given text.

        Args:
            ref_codes: optional [k, T_ref] reference audio codes for speaker conditioning.
                       Requires audio_start_id and audio_end_id to be provided.
        Returns:
            codes: LongTensor [k, n_frames]
        """
        device = next(self.parameters()).device
        self.eval()

        text_vocab = self.llm.config.vocab_size
        embed_tokens = self.llm.model.embed_tokens

        prompt_embeds_list = []

        # --- Optional reference block: <audio_start> ref_codes <audio_end> ---
        if ref_codes is not None and audio_start_id is not None and audio_end_id is not None:
            ref_codes = ref_codes.to(device)
            T_ref     = ref_codes.shape[1]

            # <audio_start> embedding (text token)
            start_t = torch.tensor([[audio_start_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(start_t.clamp(0, text_vocab - 1)))  # [1, 1, H]

            # Interleaved ref code embeddings
            ref_tokens = ref_codes.T.reshape(-1)  # [T_ref*k]
            for pos, code in enumerate(ref_tokens):
                cb_idx = pos % self.k
                idx    = code.clamp(0, self.cfg.codebook_size - 1).unsqueeze(0)  # [1]
                emb    = self.audio_embeds[cb_idx](idx).unsqueeze(0)             # [1, 1, H]
                prompt_embeds_list.append(emb)

            # <audio_end> embedding
            end_t = torch.tensor([[audio_end_id]], dtype=torch.long, device=device)
            prompt_embeds_list.append(embed_tokens(end_t.clamp(0, text_vocab - 1)))  # [1, 1, H]

        # --- Text + sep ---
        text_ids  = tokenizer.encode(text, add_special_tokens=False)
        full_ids  = text_ids + [self.audio_sep_id]
        id_tensor = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
        prompt_embeds_list.append(embed_tokens(id_tensor.clamp(0, text_vocab - 1)))  # [1, L, H]

        # Concatenate all prompt embeddings and seed KV cache
        llm_dtype     = next(self.llm.parameters()).dtype
        prompt_embeds = torch.cat(prompt_embeds_list, dim=1).to(llm_dtype)  # [1, L_total, H]
        out      = self.llm.model(inputs_embeds=prompt_embeds, use_cache=True)
        hidden   = out.last_hidden_state   # [1, L_total, H]
        past_kv  = out.past_key_values

        all_codes: list = []
        audio_pos  = 0  # counts generated audio tokens
        n_frames   = 0  # completed frames

        while n_frames < max_audio_frames:
            cb_idx = audio_pos % self.k

            # Predict next code from last hidden state
            h      = hidden[:, -1:, :]                               # [1, 1, H]
            logits = self.audio_heads[cb_idx](h.float()).squeeze(1)  # [1, vocab]

            # Suppress EOS until min_audio_frames have been generated
            if cb_idx == 0 and n_frames < min_audio_frames:
                logits[:, self.AUDIO_EOS] = float("-inf")

            next_code = _sample(logits, temperature, top_k, top_p)  # [1]

            # cb0 can predict AUDIO_EOS (index 2048) → stop generation
            if cb_idx == 0 and next_code.item() == self.AUDIO_EOS:
                break

            all_codes.append(next_code.item())
            audio_pos += 1
            if audio_pos % self.k == 0:
                n_frames += 1

            # Clamp to valid embedding index before lookup
            embed_idx = next_code.clamp(0, self.cfg.codebook_size - 1)
            embed = self.audio_embeds[cb_idx](embed_idx.unsqueeze(0))  # [1, 1, H]
            embed = embed.to(hidden.dtype)
            out     = self.llm.model(inputs_embeds=embed, past_key_values=past_kv, use_cache=True)
            hidden  = out.last_hidden_state   # [1, 1, H]
            past_kv = out.past_key_values

        # Trim to complete frames and reshape to [k, n_frames]
        complete = (len(all_codes) // self.k) * self.k
        if complete == 0:
            return torch.zeros(self.k, 0, dtype=torch.long)
        codes = torch.tensor(all_codes[:complete], dtype=torch.long)
        codes = codes.reshape(complete // self.k, self.k).T  # [k, n_frames]
        return codes


def _sample(
    logits: torch.Tensor,   # [1, vocab]
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.LongTensor:
    """Temperature + top-k + top-p sampling. Returns scalar LongTensor."""
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens that push cumulative prob above top_p
        remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)  # [1]
