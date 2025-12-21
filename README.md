
# RAGTruth Dataset

## Dataset Description

### Dataset Summary
The RAGTruth dataset is designed for evaluating hallucinations in text generation models, particularly in retrieval-augmented generation (RAG) contexts. It contains examples of model outputs along with expert annotations indicating whether the outputs contain hallucinations.

### Dataset Structure
Each example contains:
- A query/question
- Context passages
- Model output
- Hallucination labels (evident conflict and/or baseless information)
- Quality assessment
- Model metadata (name, temperature)

## Dataset Statistics

### Train Split

- Total examples: 15090
- Examples with hallucinations: 6721

#### Hallucination Label Distribution
- Evident Conflict: 3389
- Baseless Info: 4945
- Both types: 1613

#### Quality Label Distribution
- good: 14942
- truncated: 28
- incorrect_refusal: 120

### Test Split

- Total examples: 2700
- Examples with hallucinations: 943

#### Hallucination Label Distribution
- Evident Conflict: 469
- Baseless Info: 638
- Both types: 164

#### Quality Label Distribution
- good: 2675
- incorrect_refusal: 24
- truncated: 1



## Dataset Creation

### Annotations
Annotations were created by expert reviewers who identified two types of hallucinations:
- Evident Conflict: Information that directly contradicts the provided context
- Baseless Information: Information not supported by the context

### Licensing Information
This dataset is released under the MIT License.
