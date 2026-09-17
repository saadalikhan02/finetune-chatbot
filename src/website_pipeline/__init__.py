"""Website crawling and grounded-dataset-generation pipeline for the
Technyx Systems corporate chatbot.

This package is deliberately separate from ``corporate_chatbot`` (the
model-training package): it produces JSONL datasets in that project's
expected format, but the crawling/extraction concerns here have nothing to
do with training itself.
"""
