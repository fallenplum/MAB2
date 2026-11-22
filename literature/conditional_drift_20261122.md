# Literature update: conditional graph-credit revalidation

## Primary references checked on 2026-11-22

- Javdani, Chen, Karbasi, Krause, Bagnell, Srinivasa. “Near Optimal Bayesian Active Learning for Decision Making.” AISTATS 2014. https://proceedings.mlr.press/v33/javdani14.pdf
  - Decision-region determination and cost-aware tests are established. A new method cannot claim decision-region reduction or paid-test scheduling by itself.
- Wang et al. “AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration.” ACL 2025. https://aclanthology.org/2025.acl-long.1170/
  - Dynamic graph/agent elimination with token and task-quality evaluation is a direct efficiency neighbor.
- Zhang et al. “Understanding the Information Propagation Effects of Communication Topologies in LLM-Based Multi-Agent Systems.” EMNLP 2025. https://aclanthology.org/2025.emnlp-main.623/
  - Communication topology and propagation effects are already studied with task quality and cost.
- You et al. “MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based Multi-Agent Communication.” EMNLP 2026. https://aclanthology.org/2026.emnlp-main.2179/
  - Runtime message-instance pruning in fixed graphs is a strong cost baseline.
- Chen et al. “AgentSlimming: Towards Efficient and Cost-Aware Multi-Agent Systems.” ACL 2026. https://aclanthology.org/2026.acl-long.1387/
  - Graph-structured workflow compression with quality-preserving acceptance rules is a recent neighbor.

## Novelty boundary

The defensible gap is narrower. Existing work studies graph pruning, message pruning, topology effects, and generic decision-region testing. MAB2 can claim a contribution only if it demonstrates that graph edits change the validity of previously paid credit certificates, and that a conditional-channel-aware revalidation policy detects or repairs this failure under a fixed feedback budget. The conditional-channel failure must be measured on new model outputs and separated from transcript reuse, source reweighting, and generic decision-region ranking.
