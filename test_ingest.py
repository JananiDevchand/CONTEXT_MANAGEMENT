from file_ingestion import ingest_markdown

USER_ID = "11111111-1111-1111-1111-111111111111"


markdown = """
# Transformer Architecture

Transformers are a class of neural network architectures designed to handle
sequential data by relying entirely on self-attention mechanisms rather than
recurrent or convolutional layers.

## Self-Attention
Self-attention allows the model to weigh the importance of different tokens
in a sequence relative to each other. This enables parallel computation and
better handling of long-range dependencies.

## Encoder
The encoder consists of multiple identical layers. Each layer contains a
multi-head self-attention mechanism followed by a feed-forward neural network.

## Decoder
The decoder also contains multiple layers and includes an additional
encoder-decoder attention mechanism that allows it to focus on relevant
parts of the input sequence during generation.

## Applications
Transformers are widely used in natural language processing tasks such as
machine translation, text summarization, question answering, and language
modeling. They form the backbone of modern large language models.
"""


file_id = ingest_markdown(
    user_id=USER_ID,
    filename="transformer_notes.md",
    markdown=markdown
)

print("Stored file ID:", file_id)
