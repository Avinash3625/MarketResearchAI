# NER Labeling Strategy

## Overview

This document outlines the labeling strategy for Named Entity Recognition (NER) training data used in MarketResearchAI.

## Entity Schema

### Primary Entity Types (BIO Format)

| Tag | Description | Examples |
|-----|-------------|----------|
| `B-PER` | Beginning of Person | "Elon", "Warren" |
| `I-PER` | Inside of Person | "Musk", "Buffett" |
| `B-ORG` | Beginning of Organization | "Apple", "Microsoft" |
| `I-ORG` | Inside of Organization | "Inc", "Corp" |
| `B-LOC` | Beginning of Location | "New", "California" |
| `I-LOC` | Inside of Location | "York", "City" |
| `B-MISC` | Beginning of Miscellaneous | "GPT-4", "iPhone" |
| `I-MISC` | Inside of Miscellaneous | (continuations) |
| `O` | Outside (non-entity) | "the", "is", "and" |

### BIO Tagging Convention
- **B-** (Beginning): First token of a named entity
- **I-** (Inside): Subsequent tokens within the same entity
- **O** (Outside): Tokens that are not part of any entity

## Data Format

### JSONL Structure
```json
{
    "tokens": ["Apple", "Inc", ".", "is", "in", "California", "."],
    "labels": ["B-ORG", "I-ORG", "O", "O", "O", "B-LOC", "O"]
}
```

### Requirements
- One JSON object per line
- `tokens`: Pre-tokenized list of words
- `labels`: Corresponding BIO labels (same length as tokens)

## Quality Control (QC)

### Annotation Guidelines

1. **Consistency**: Same entities should be labeled identically across documents
2. **Completeness**: All tokens in a multi-word entity must be labeled
3. **Boundary Precision**: Entity boundaries must be accurate

### QC Process

1. **Inter-Annotator Agreement (IAA)**
   - Calculate Cohen's kappa for label agreement
   - Target: κ > 0.8 for production data

2. **Review Workflow**
   - Initial annotation by annotator
   - Review by second annotator
   - Adjudication for disagreements

3. **Automated Checks**
   - Validate BIO sequence (no I- without preceding B-)
   - Check token/label count match
   - Flag ambiguous cases

### Quality Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| IAA Score | > 0.8 | Cohen's Kappa |
| Annotation Coverage | 100% | All docs labeled |
| B/I Sequence Validity | 100% | Valid BIO sequences |

## Export Formats

### 1. JSONL (Primary)
```json
{"tokens": [...], "labels": [...]}
```

### 2. CoNLL Format
```
Apple   B-ORG
Inc     I-ORG
.       O
```

### 3. Spacy Format
```python
[("Apple Inc. is in California.", {"entities": [(0, 10, "ORG"), (17, 27, "LOC")]})]
```

## Labeling Tools

### Recommended
- **Label Studio**: Web-based, supports team collaboration
- **Prodigy**: Commercial, efficient for NER
- **Doccano**: Open-source, simple interface

### Export Scripts

Convert between formats using:
```bash
python scripts/convert_labels.py --input data.jsonl --output data.conll --format conll
```

## Handling Edge Cases

### Nested Entities
- Label outermost entity only (limitation of BIO)
- Document nested cases for potential future handling

### Abbreviations
- `B-ORG` for acronyms like "IBM", "AWS"

### Punctuation
- Attached punctuation: Include with entity
- Separated punctuation: Label as `O`

## Data Split Recommendations

| Split | Percentage | Purpose |
|-------|------------|---------|
| Train | 80% | Model training |
| Validation | 10% | Hyperparameter tuning |
| Test | 10% | Final evaluation |

## Version Control

- Store labeled data in version control
- Use semantic versioning for dataset releases
- Document changes between versions

## Limitations

1. **BIO Scheme**: Does not support nested entities
2. **Pre-tokenization**: Tokenization affects entity boundaries
3. **Domain Adaptation**: Market research entities may need custom types

## Future Improvements

1. Add domain-specific entity types (PRODUCT, METRIC, TREND)
2. Implement BIOES tagging for better boundary detection
3. Support for relation extraction annotations
