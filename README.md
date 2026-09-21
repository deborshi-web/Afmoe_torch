# Afmoe

AFMOE (Adaptive Feature Mixture-of-Experts) is a neural network architecture designed to improve model performance by dynamically selecting and combining specialized expert networks for different input features or patterns.



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

Instead of sending every input through the same computational pathway, AFMOE uses a gating mechanism to determine which experts are most relevant to a particular input. The selected experts process the input, and their outputs are adaptively weighted and combined to produce the final prediction.


A typical AFMOE pipeline can be represented as:

*Input → Feature Extraction → Adaptive Gating → Expert Networks → Weighted Feature Fusion → Output*

The main idea is to allow different experts to specialize in different characteristics of the data while the gating network learns when and how much each expert should contribute. This can provide greater flexibility than a conventional single-network architecture and can be useful for complex tasks involving diverse or heterogeneous patterns.


**Key characteristics**:

**Adaptive routing**: dynamically chooses relevant experts.
Expert specialization: different experts can learn different feature representations.

**Feature fusion**: combines expert outputs using learned weights.

**Scalability**: additional experts can be introduced for more specialized representations.

**Potential efficiency**: only a subset of experts may need to be activated for each input.


AFMOE can be adapted to applications such as computer vision, natural-language processing, time-series analysis, multimodal learning, and other machine-learning problems where different inputs may benefit from different learned representations.
