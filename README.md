# Afmoe_Tensorflow

A Mixture-of-Experts (MoE) model is an AI architecture that splits a large neural network into smaller specialized sub-networks called experts, activating only the most relevant ones for each piece of data.



```mermaid



classDiagram
    direction TB

    %% Inheritance
    PreTrainedModel <|-- AfmoePreTrainedModel
    AfmoePreTrainedModel <|-- AfmoeModel
    AfmoePreTrainedModel <|-- AfmoeForCausalLM
    GradientCheckpointingLayer <|-- AfmoeDecoderLayer
    LlamaAttention <|-- AfmoeAttention

    %% Main classes & members
    class AfmoePreTrainedModel {
      +config: AfmoeConfig
      +_init_weights(module)
    }

    class AfmoeModel {
      +embed_tokens: nn.Embedding
      +layers: "ModuleList[AfmoeDecoderLayer]"
      +norm: AfmoeRMSNorm
      +rotary_emb: AfmoeRotaryEmbedding
      +forward(...)
    }

    class AfmoeForCausalLM {
      +model: AfmoeModel
      +lm_head: nn.Linear
      +forward(...)
    }

    class AfmoeDecoderLayer {
      +self_attn: AfmoeAttention
      +input_layernorm: AfmoeRMSNorm
      +post_attention_layernorm: AfmoeRMSNorm
      +pre_mlp_layernorm: AfmoeRMSNorm
      +post_mlp_layernorm: AfmoeRMSNorm
      +mlp: AfmoeMLP | AfmoeSparseMoeBlock
      +forward(...)
    }

    class AfmoeAttention {
      +q_norm: AfmoeRMSNorm
      +k_norm: AfmoeRMSNorm
      +gate_proj: nn.Linear
      +forward(...)
    }

    class AfmoeSparseMoeBlock {
      +router: AfmoeTokenChoiceRouter
      +shared_experts: AfmoeMLP
      +experts: AfmoeExperts
      +expert_bias: Parameter
      +forward(hidden_states)
    }

    class AfmoeTokenChoiceRouter {
      +gate: nn.Linear
      +top_k: int
      +forward(hidden_states, expert_bias) -> router_logits, top_scores, selected_experts
    }

    class AfmoeMLP
    class AfmoeExperts
    class AfmoeRMSNorm
    class AfmoeRotaryEmbedding

    %% Composition / usage arrows
    AfmoeModel --> AfmoeDecoderLayer : layers (ModuleList)
    AfmoeModel --> AfmoeRotaryEmbedding : rotary_emb
    AfmoeModel --> nn.Embedding : embed_tokens
    AfmoeForCausalLM --> AfmoeModel : model
    AfmoeForCausalLM --> nn.Linear : lm_head

    AfmoeDecoderLayer --> AfmoeAttention : self_attn
    AfmoeDecoderLayer --> AfmoeRMSNorm : layernorms
    AfmoeDecoderLayer --> AfmoeMLP : dense MLP (if dense)
    AfmoeDecoderLayer --> AfmoeSparseMoeBlock : MoE block (if moe_enabled)

    AfmoeSparseMoeBlock --> AfmoeTokenChoiceRouter : router
    AfmoeSparseMoeBlock --> AfmoeMLP : shared_experts
    AfmoeSparseMoeBlock --> AfmoeExperts : experts
```
    
