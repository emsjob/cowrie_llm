import os
#To be added for LLM
if os.environ["COWRIE_USE_LLM"].lower() == "true":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from accelerate import init_empty_weights, dispatch_model
    import torch

import json

RESPONSE_PATH = "/cowrie/cowrie-git/src/model"

PROMPTS_PATH = "/cowrie/cowrie-git/src/model/prompts"

with open(f"{RESPONSE_PATH}/cmd_lookup.json", "r") as f:
    LOOKUPS = json.load(f)

class LLM:
    def __init__(self, model_name="google/codegemma-7b-it"):
        with open(f"{RESPONSE_PATH}/token.txt", "r") as f:
            token = f.read().rstrip()

        self.profile = self.get_profile()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        #quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

        self.model = dispatch_model(model, device_map="auto", quantization_config=quantization_config)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map="auto", quantization_config=quantization_config)

    def get_profile(self):
        with open(PROMPTS_PATH+"/profile.txt", "r") as prompt_file:
            profile = prompt_file.read()
        return profile

    def get_examples(self, cmd):
        with open(PROMPTS_PATH+f"/ex_{cmd}.json", "r") as ex_file:
            examples = json.load(ex_file)
        return examples

    def create_messages(self, base_prompt, cmd):
        answer = LOOKUPS[cmd]
        messages = [
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": cmd}
        ]
        return messages

    def generate_response(self, cmd):
        base_prompt = f"You are Linux OS terminal for a server containing sensitive patient data. "+\
            "Your personality is: You are a Linux OS terminal. You act and respond exactly as a Linux terminal. "+\
            "You will respond to all commands just as a Linux terminal would. " +\
            "You can only respond to user inputs and you must not write any commands on your own. " +\
            "You must not in any case have a conversation with user as a chatbot and must not explain your output and do not repeat commands user inputs. " +\
            "Do not explain to user what they are seeing. Only respond as Linux terminal. "+\
            "You will need to make up realistic answers to the command, as they would be returned by a real linux terminal for a hospital server. "+\
            "It is very important that you do not name files and directiories file1.txt file2.txt file3.txt or similarly, rather create plausible file names for a real terminal with patient data.\n\n"+\
            "{cmd}"

        messages = self.create_messages(base_prompt, cmd)
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        len_chat = tokenized_chat.shape[1]
        outputs = self.model.generate(tokenized_chat, max_new_tokens=500)
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        return response
    
    def generate_from_messages(self, messages, max_new_tokens=100):
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        print("tokenized chat:")
        print(tokenized_chat)
        len_chat = tokenized_chat.shape[1]
        outputs = self.model.generate(tokenized_chat, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        return response

    def generate_ls_response(self, cwd):
        def format_q(cmd, cwd):
            return f"Command: {cmd}\nCurrent directory: {cwd}"

        #Maybe we should load all these by initialisation
        examples = self.get_examples("ls")
        ex_q = [format_q(ex["cmd"], ex["cwd"]) for ex in examples]
        ex_a = [ex["response"] for ex in examples]

        messages = [{"role":"user", "content":self.profile},
                    {"role":"model", "content":""}]
        for i in range(len(examples)):
            messages.append({"role":"user", "content":ex_q[i]})
            messages.append({"role":"model", "content":ex_a[i]})
        
        messages.append({"role":"user", "content":format_q("ls", cwd)})

        return self.generate_from_messages(messages)

    def generate_lscpu_response(self):
        profile = self.get_profile()

        return "Makeshift lscpu response"
    
class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            return "Something generated by a LLM"
        return func