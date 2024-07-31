import os
if os.environ["COWRIE_USE_LLM"].lower() == "true":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    import torch
import json
import numpy as np
import time
import re

RESPONSE_PATH = "/cowrie/cowrie-git/src/model"
PROMPTS_PATH = "/cowrie/cowrie-git/src/model/prompts"

TEMPLATE_TOKEN = "<unk>"
TEMPLATE_TOKEN_ID = 0
SYSTEM_ROLE_AVAILABLE = True

#MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

with open(f"{RESPONSE_PATH}/cmd_lookup.json", "r") as f:
    LOOKUPS = json.load(f)

class LLM:
#region base
    def __init__(self, model_name=MODEL_NAME):
        with open(f"{RESPONSE_PATH}/token.txt", "r") as f:
            token = f.read().rstrip()

        self.profile = self.get_profile()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.num_connections = None

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=token, device_map="auto", quantization_config=quantization_config)

    def get_profile(self):
        with open(PROMPTS_PATH+"/large_profile.txt", "r") as prompt_file:
            profile = prompt_file.read()
        return profile

    def get_examples(self, cmd):
        with open(PROMPTS_PATH+f"/ex_{cmd}.json", "r") as ex_file:
            examples = json.load(ex_file)
        return examples

    def generate_from_messages(self, messages, max_new_tokens=100):
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        print("prompt:")
        print(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        len_chat = tokenized_chat.shape[1]
        gen_start_time = time.time()
        outputs = self.model.generate(tokenized_chat, max_new_tokens=max_new_tokens, do_sample=True, num_beams=1, top_k=5, temperature=0.6)
        gen_end_time = time.time()
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        print(f"LLM GENERATION TIME: {gen_end_time - gen_start_time}")
        return response
#endregion

#region general response
    def generate_general_response(self, cmd, extra_info=None):
        base_prompt = self.profile
        examples = self.get_examples(cmd)
        resp_str = examples[0]["response"]
        num_tokens = 1.2 * len(self.tokenizer.tokenize(resp_str))
        if len(examples) > 0:
            base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the {cmd} command. Do not just copy the example directly, rather adjust it appropriately.'
            for i in range(len(examples)):
                base_prompt = base_prompt+f"\n\nExample {i+1}:\n"+examples[i]["response"]

        if extra_info:
            basse_prompt = base_prompt + extra_info
        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"assistant", "content":""}
                ]
        messages.append({"role":"user", "content":f"COMMAND: {cmd}"})

        return self.generate_from_messages(messages, max_new_tokens=num_tokens)
#endregion

#region ls
    def format_ls_q(self, cmd, cwd):
        return f"Command: {cmd}\nPath: {cwd}"

    def generate_ls_response(self, cwd, history=None):
        #Maybe we should load all these by initialisation
        examples = self.get_examples("ls")
        ex_q = [self.format_ls_q(ex["cmd"], ex["cwd"]) for ex in examples]
        ex_a = [ex["response"] for ex in examples]

        base_prompt = self.profile + f"\n\nMake sure that you generate more files in deeper directories. The following {len(examples)} interactions are examples of the responses to the ls command."

        messages = [{"role":"system", "content":base_prompt}]
        for i in range(len(examples)):
            messages.append({"role":"user", "content":ex_q[i]})
            messages.append({"role":"assistant", "content":ex_a[i]})

        if history:
            messages.append({"role":"system", "content":f"The following {len(history)} interactions are your past interactions with the user. Try to stay consistent with them."})
            for event in history:
                messages.append({"role":"user", "content":self.format_ls_q(event["cmd"], event["path"])})
                messages.append({"role":"assistant", "content":event["response"]})
        
        messages.append({"role":"user", "content":self.format_ls_q("ls", cwd)})

        return self.generate_from_messages(messages)
#endregion

#region file contents
    def format_file_q(self, path):
        return f"File contents in: {path}"

    def generate_file_contents(self, path):
        examples = self.get_examples("file_contents")

        ex_q = [self.format_file_q(ex["path"]) for ex in examples]
        ex_a = [ex["response"] for ex in examples]

        base_prompt = self.profile + f"\n\nThe following {len(examples)} interactions are examples for content in different types of files."

        messages = [{"role":"system", "content":base_prompt}]
        for i in range(len(examples)):
            messages.append({"role":"user", "content":ex_q[i]})
            messages.append({"role":"assistant", "content":ex_a[i]})
        

        messages.append({"role":"user", "content":self.format_file_q(path)})

        return self.generate_from_messages(messages, max_new_tokens=300)
#endregion

#region ifconfig
    def generate_ifconfig_response(self, use_template=False):
        return self.generate_general_response("ifconfig")
#endregion

#region netstat
    def generate_netstat_response(self, use_template=False):
        return self.generate_general_response("netstat")
#endregion

#region free
    def generate_free_response(self):
        response = self.generate_general_response("free").split()

        mem_start_index = response.index("Mem:") + 1
        swap_start_index = response.index("Swap:") + 1 if "Swap:" in response else None

        mem_values = response[mem_start_index:mem_start_index + 6]
        if swap_start_index:
            swap_values = response[swap_start_index:swap_start_index + 3]
        else:
            swap_values = ["0", "0", "0"]

        values = {
            "MemTotal": mem_values[0],
            "calc_total_used": str(int(mem_values[0]) - int(mem_values[2]) - int(mem_values[4])),
            "MemFree": mem_values[2],
            "Shmem": mem_values[3],
            "calc_total_buffers_and_cache": str(int(mem_values[4])),
            "MemAvailable": mem_values[5],
            "SwapTotal": swap_values[0],
            "calc_swap_used": str(int(swap_values[0]) - int(swap_values[2])),
            "SwapFree": swap_values[2],
        }
        template = """              total        used        free      shared  buff/cache   available
Mem:{MemTotal:>15}{calc_total_used:>12}{MemFree:>12}{Shmem:>12}{calc_total_buffers_and_cache:>12}{MemAvailable:>12}
Swap:{SwapTotal:>14}{calc_swap_used:>12}{SwapFree:>12}
"""
        filled_template = template.format(**values).rstrip()
        return filled_template
#endregion

#region last
    def generate_last_response(self):
        with open("/cowrie/cowrie-git/honeyfs/etc/passwd", "r") as passwd_file:
            users = passwd_file.read()
        response = self.generate_general_response("last", extra_info="\nHere is a list of users to use in the leftmost column: {users}\n")
        return response
#endregion

#region staticcmds
    def generate_lscpu_response(self):
        response = self.generate_general_response("lscpu", extra_info="\nChange the values to reasonable ones considering the size of the system.\n").split("\n")
        values = {}
        for line in response:
            if line.strip():
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()

        template = """Architecture:          {Architecture}
CPU op-mode(s):        {CPU op-mode(s)}
Byte Order:            {Byte Order}
CPU(s):                {CPU(s)}
On-line CPU(s) list:   {On-line CPU(s) list}
Thread(s) per core:    {Thread(s) per core}
Core(s) per socket:    {Core(s) per socket}
Socket(s):             {Socket(s)}
NUMA node(s):          {NUMA node(s)}
Vendor ID:             {Vendor ID}
CPU family:            {CPU family}
Model:                 {Model}
Stepping:              {Stepping}
CPU MHz:               {CPU MHz}
BogoMIPS:              {BogoMIPS}
Hypervisor vendor:     {Hypervisor vendor}
Virtualization type:   {Virtualization type}
L1d cache:             {L1d cache}
L1i cache:             {L1i cache}
L2 cache:              {L2 cache}
NUMA node0 CPU(s):     {NUMA node0 CPU(s)}
        """

        filled_template = template.format(**values).rstrip()
        return filled_template

    def generate_nproc_response(self):
        return self.generate_general_response("nproc", extra_info="\nA large system have 16 or 32 and a small system 2 or 4.\n")

    def generate_df_response(self):
        response = self.generate_general_response("df", extra_info="\nDo not use anything other than the english language.\n").split()

        response = response[6:]
        columns_per_row = 6
        num_rows = len(response) // columns_per_row
        filesystems = []
        for i in range(1, num_rows):
            start_index = i * columns_per_row + 1
            filesystem_info = {
                "Filesystem": response[start_index],
                "Size": response[start_index + 1],
                "Used": response[start_index + 2],
                "Avail": response[start_index + 3],
                "Use%": response[start_index + 4],
                "Mounted_on": " ".join(response[start_index + 5:start_index + 6])
            }
            filesystems.append(filesystem_info)

        template = """Filesystem                                               Size  Used Avail Use% Mounted on
{rows}
"""

        rows = ""
        for fs in filesystems:
            row = "{Filesystem:<55}{Size:>5} {Used:>5} {Avail:>5} {Use%:>5} {Mounted_on:<}\n".format(**fs)
            rows += row

        filled_template = template.format(rows=rows).rstrip()
        return filled_template
#endregion

#region hostname
    def generate_host_name(self):
        messages = [{"role":"system", "content":self.profile},
                    {"role":"user", "content":"Respond with a short and creative host name for this system, without spaces. Do not simply name it 'host' or something similar but consider the actual profile of the system. Respond only with the string representing the hostname."}]
        return self.generate_from_messages(messages)
#endregion

#region add users
    def add_users(self, passwd):
        base_prompt = self.profile + "\nYou need to extend the base /etc/passwd file with realistic users which would be added to the system. Make sure that users you add have their home directory inside the /home directory, like the admin user. The base /etc/passwd file which you should extend is:\n" + passwd

        messages = [{"role":"system", "content":base_prompt},
                    {"role":"user", "content":"cat /etc/passwd"}]

        return self.generate_from_messages(messages, max_new_tokens=1000)
#endregion

#region support-classes
class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            return "Something generated by a LLM"
        return func
    
class NewWordSC(StoppingCriteria):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        lasts = input_ids[:, -1]
        res = torch.zeros_like(lasts, dtype=torch.bool)
        for i in range(lasts.shape[0]):
            decoded = self.tokenizer.decode(lasts[i])
            if " " in decoded:
                res[i] = True
            elif "\n" in decoded:
                res[i] = True
            elif "\t" in decoded:
                res[i] = True
            elif decoded == "":
              res[i] = True
        return res
#endregion

#region ServiceLLM
import multiprocessing as mp
from queue import Empty
import threading
import uuid

class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instances.get(cls) is None:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ServiceLLM(metaclass=SingletonMeta):
    def __init__(self, model_name=MODEL_NAME):
        print("Starting LLM service")
        self.model_name=model_name

        self.manager = mp.Manager()

        self.queue = mp.Queue()
        self.result_pipes = self.manager.dict()

        self.worker = mp.Process(target=self._process_requests)
        self.worker.start()

    def _process_requests(self):
        print("starting worker")
        llm = LLM(model_name=self.model_name)
        print("starting loop")
        while True:
            try:
                request_id, method_name, args, kwargs = self.queue.get(timeout=1)
                result = getattr(llm, method_name)(*args, **kwargs)
                result_pipe = self.result_pipes.pop(request_id, None)

                if result_pipe:
                    result_pipe.send(result)
                    result_pipe.close()
                else:
                    print("could not find pipe")
            except Empty:
                continue

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            parent_conn, child_conn = mp.Pipe()

            request_id = str(uuid.uuid4())
            self.result_pipes[request_id] = child_conn
            self.queue.put((request_id, attr, args, kwargs))
            result = parent_conn.recv()
            parent_conn.close()
            return result
        return func
#endregion