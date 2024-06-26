# VLMs Bias Explorer

This repo aims to explore social and computational bias in VLMs, especially CLIP and those derived from it. There is a package that can generate embeddings, evaluate over a classification task and save a report of the performance. More instructions e-mail me at [lmceschini@inf.ufrgs.br](lmceschini@inf.ufrgs.br).

## Possible Experiments

By leveraging the capabilities of multiple backbones and data sources, we can think of a variety of possible experiments.

- Impact of datasources
- Impact of backbones

## Preliminary Studies

We will start by handpicking some model backbones and datasources.

### Model Backbones

- ViT-L-14
- ViT-B-32

### Datasources

- openai
- laion400m_e31
- laion2b_s32b_b82k
- datacomp_xl_s13b_b90k
- commonpool_xl_s13b_b90k
