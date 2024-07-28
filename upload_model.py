from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, Repository

# Define your model directory and repository name
model_dir = "restful-llama-instruct"  # Replace with the actual path to your model directory
repo_name = "the1ullneversee/Restful-Llama-3-7b-Instruct"

model = AutoModel.from_pretrained("restful-llama")
tokenizer = AutoTokenizer.from_pretrained("restful-llama")

repo = Repository(local_dir=model_dir, clone_from=repo_name)

# Save the model and tokenizer to the local repository
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

repo.push_to_hub()