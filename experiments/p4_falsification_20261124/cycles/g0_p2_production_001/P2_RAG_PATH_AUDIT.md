# P2 RAG path audit

The frozen `retrieval_augmented_generation` branch in `p2_service_runner.py` builds a prompt from the two row facts and sets `retrieval_calls = 2`. It does not call a retriever, emit evidence IDs, store ranked documents, or record retrieval latency and cost from an external corpus.

The current P4 adapter therefore does not claim T06. The four Qwen rows exercise the graph renderer and broker only. A true T06 run requires a versioned public corpus, query and ranking logs, an identical information source for all methods, and a real retrieval hook. No such corpus is present in the frozen P2 input package.

Status `T06_OPEN_NO_TRUE_RETRIEVER`.
