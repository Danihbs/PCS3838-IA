# Circuit Graph Extraction Pipeline

## Overview

The pipeline combines classical computer vision with a vision-language
model fallback to answer questions about logic circuit diagrams.

Pipeline:

Image
  ↓
Gate detection
  ↓
Wire skeletonization
  ↓
Circuit graph extraction
  ↓
External input identification
  ↓
Boolean graph simulation

Qwen2.5-VL is used only as a fallback when deterministic extraction fails.

---

## Version 3 — Graph Extraction

Version 3 introduced deterministic circuit graph extraction using OpenCV
and NetworkX.

### Main improvements

- Gate detection based on internal contours
- Gate classification for AND, OR, NOT, NAND, NOR, XOR and XNOR
- Wire extraction using skeletonization
- Gate-to-gate connection reconstruction
- Directed circuit graph generation
- Deterministic gate counting
- Boolean graph simulation

### Limitation

External circuit inputs were assigned names (`x0`, `x1`, ...)
according to their spatial order.

This does not necessarily match the labels shown in the original image.

As a result, the circuit topology could be correct while Boolean
simulation used the wrong values for the input wires.

Observed validation results:

- Gate counting: ~96%
- Graph validity: 100/100 tested images
- Output simulation: ~60%

---

## Version 3.1 — OCR-based Input Mapping

Version 3.1 addresses the input-mapping problem by reading the actual
input labels from the image.

### New pipeline stage

Image labels
  ↓
Character blob detection
  ↓
Blob clustering
  ↓
Tesseract OCR
  ↓
Label normalization
  ↓
Endpoint-to-label matching
  ↓
Circuit graph

Instead of assigning arbitrary positional names, unresolved external
wire endpoints are matched with labels such as:

x0, x1, x7, x12, ...

### Additional fixes

The implementation treats every unresolved wire endpoint as an
independent input slot.

This prevents multiple external inputs connected to the same gate from
overwriting each other during OCR matching.

### Fallback behavior

If OCR is unavailable or no nearby label can be confidently matched,
the pipeline falls back to the sequential naming strategy used in v3.