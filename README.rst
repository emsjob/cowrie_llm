Cowrie LLM
######

An LLM enhanced Cowrie shell.

Built on the original Cowrie shell (https://github.com/cowrie/cowrie) with an LLM integrated into it.

The honeypot is easily adaptable and the LLM can generate different types of systems by modifying the following file:
``src/model/prompts/profile.txt``

Run Instructions:

1. Install stuff for GPU:
``chmod +x gpu_configuration.sh && ./gpu_configuration.sh``

2.1 Build Docker with:
``docker build --build-arg HUGGINGFACE_TOKEN=$(cat src/model/token.txt) --build-arg HUGGINGFACE_USERNAME="your_huggingface_username" -f docker/Dockerfile -t cowrie:latest .``

2.2 Alternatively, if you want to build quicker by not having a real LLM, use:
``docker build -f docker/Dockerfile_no_llm -t cowrie:latest .``

3. Run Docker with:
``docker run --gpus all -e COWRIE_USE_LLM=true -p 2222:2222 cowrie:latest``

