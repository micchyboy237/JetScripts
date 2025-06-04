from transformers import SpanBERTTokenizer, SpanBERTForQuestionAnswering

tokenizer = SpanBERTTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
model = SpanBERTForQuestionAnswering.from_pretrained(
    "SpanBERT/spanbert-large-cased")

question = "What is the capital of France?"
context = "The capital of France is Paris."
inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_idx = outputs.start_logits.argmax()
end_idx = outputs.end_logits.argmax()
answer = tokenizer.decode(inputs.input_ids[0][start_idx:end_idx+1])
