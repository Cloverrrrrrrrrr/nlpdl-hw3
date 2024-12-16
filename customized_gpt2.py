from turtle import forward
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel


class CustomizedGPT2Attention(GPT2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        past_key_value: Tuple of (key, value) from previous forward passes. 
        If None, the attention will be computed from scratch.
        """
        
        

        past_key, past_value = past_key_value
        if past_key is None and past_value is None:           
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)  # Each of them has shape (batch_size, seq_len, dim)
            query = self._split_heads(query, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            key = self._split_heads(key, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            value = self._split_heads(value, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            
        else:
            h = hidden_states[:, -1, :].unsqueeze(1)  
            
            
            query, key, value = self.c_attn(h).split(self.split_size, dim=2)  # Each of them has shape (batch_size, seq_len, dim)
            query = self._split_heads(query, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            key = self._split_heads(key, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            value = self._split_heads(value, self.num_heads, self.head_dim)  # [batch_size, num_heads, seq_len, head_dim]
            key = torch.cat((past_key, key), dim=-2)  # Concatenate the new key to the old one
            value = torch.cat((past_value, value), dim=-2)  # Concatenate the new value to the old one
            

        
    

        # Self-attention mechanism
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)  # [batch_size, seq_len, dim]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        # Return the attention output and the updated key-value pair for the next layer
        return attn_output, (key, value)


class CustomizedGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        self.attn = CustomizedGPT2Attention(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # Add past_key_values as input
        **kwargs,
    ):
        residual = hidden_states

        # self-attention with KV-cache support
        hidden_states = self.ln_1(hidden_states)
        
        
        

        attn_output, new_past_key_values = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
        )

        

        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states

        # feed-forward
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states, new_past_key_values


class CustomizedGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        self.h = nn.ModuleList([CustomizedGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self._attn_implementation = config._attn_implementation
        assert self._attn_implementation == 'eager', "[NLPDL ERROR] set _attn_implementation to either 'eager' or 'faster_cache' in this version"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,  # new input for past KV pairs
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        

        # Prepare input embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Prepare Attention mask.
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        hidden_states = self.drop(hidden_states)
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        # Initialize past_key_values if it's not provided
        if past_key_values is None:
            past_key_values = [(None, None)] * len(self.h)  # Initialize KV cache for each layer
        
        #print(attention_mask.type())
        
            
        # Iterate over all GPT2 layers (blocks)
        new_past_key_values = []
        for i, block in enumerate(self.h):
            
                        
            outputs, past_kv = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values[i]  # Provide KV from the previous iteration
            )
            

            hidden_states = outputs
            new_past_key_values.append(past_kv)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return hidden_states, new_past_key_values


class CustomizedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomizedGPT2Model(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,  # new input for past KV pairs
    ):  
        
        hidden_states, new_past_key_values = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values  # Pass the KV pairs to the transformer
        )
        

        # Prepare logits from last hidden state
        lm_logits = self.lm_head(hidden_states)

        return {
            'logits': lm_logits,
            'past_key_values': new_past_key_values  # Return the updated KV pairs
        }
