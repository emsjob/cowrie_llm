import os
#To be added for LLM
if os.environ["COWRIE_USE_LLM"].lower() == "true":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    import torch
import json
import numpy as np

RESPONSE_PATH = "/cowrie/cowrie-git/src/model"
PROMPTS_PATH = "/cowrie/cowrie-git/src/model/prompts"

TEMPLATE_TOKEN = "<unk>"
TEMPLATE_TOKEN_ID = 0
SYSTEM_ROLE_AVAILABLE = True


with open(f"{RESPONSE_PATH}/cmd_lookup.json", "r") as f:
    LOOKUPS = json.load(f)

class LLM:
#region base
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
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
        with open(PROMPTS_PATH+"/profile.txt", "r") as prompt_file:
            profile = prompt_file.read()
        return profile

    def get_examples(self, cmd):
        with open(PROMPTS_PATH+f"/ex_{cmd}.json", "r") as ex_file:
            examples = json.load(ex_file)
        return examples

    def generate_from_messages(self, messages, max_new_tokens=100):
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        print("prompt:")
        print(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        len_chat = tokenized_chat.shape[1]
        outputs = self.model.generate(tokenized_chat, max_new_tokens=max_new_tokens)
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        return response
#endregion

#region template
    def fill_template(self, messages, max_slot_len=20):
        tokenized_template = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        
        holes = tokenized_template == TEMPLATE_TOKEN_ID
        hole_indices = holes.nonzero()[:,1]

        stopping_criteria = StoppingCriteriaList([NewWordSC(tokenizer=self.tokenizer)])

        print("TOKENIZED TEMPLATE: ", tokenized_template)
        print("HOLES: ", holes)
        print("HOLE INDICES: ", hole_indices)

        before = tokenized_template[:, :hole_indices[0]]
        for i in range(hole_indices.shape[0]):
            hole_i = hole_indices[i]
    
            #Need to check for cutoff instead of just removing last token if we want sampling
            before = self.model.generate(before, 
                                         do_sample=False,
                                         max_new_tokens=max_slot_len,
                                         stopping_criteria=stopping_criteria,
                                         bad_words_ids=[[TEMPLATE_TOKEN_ID]])[:, :-1]
            if hole_i == hole_indices[-1]:
              tokenized_template = torch.cat([before, tokenized_template[:, hole_i+1:]], dim=1)
            else:
              before = torch.cat([before, tokenized_template[:, hole_i+1:hole_indices[i+1]]], dim=1)
        return self.tokenizer.decode(tokenized_template[0, :])
#endregion

#region ls
    def generate_ls_response(self, cwd):
        def format_q(cmd, cwd):
            return f"Command: {cmd}\nCurrent directory: {cwd}"

        #Maybe we should load all these by initialisation
        examples = self.get_examples("ls")
        ex_q = [format_q(ex["cmd"], ex["cwd"]) for ex in examples]
        ex_a = [ex["response"] for ex in examples]

        messages = [{"role":"system", "content":self.profile},
                    {"role":"assistant", "content":""}]
        for i in range(len(examples)):
            messages.append({"role":"user", "content":ex_q[i]})
            messages.append({"role":"assistant", "content":ex_a[i]})
        
        messages.append({"role":"user", "content":format_q("ls", cwd)})

        return self.generate_from_messages(messages)
#endregion

#region ifconfig
    def generate_ifconfig_response_template(self, messages):
        template = f"""
eth0      Link encap:Ethernet  HWaddr {TEMPLATE_TOKEN}  
          inet addr:{TEMPLATE_TOKEN}  Bcast:{TEMPLATE_TOKEN}  Mask:{TEMPLATE_TOKEN}
          inet6 addr: {TEMPLATE_TOKEN} Scope:Link
          UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1
          RX packets:123456 errors:0 dropped:0 overruns:0 frame:0
          TX packets:123456 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:{TEMPLATE_TOKEN} ({TEMPLATE_TOKEN} MB)  TX bytes:{TEMPLATE_TOKEN} ({TEMPLATE_TOKEN} MB)
          Interrupt:20 Memory:fa800000-fa820000 

lo        Link encap:Local Loopback  
          inet addr:{TEMPLATE_TOKEN}  Mask:{TEMPLATE_TOKEN}
          inet6 addr: ::1/128 Scope:Host
          UP LOOPBACK RUNNING  MTU:{TEMPLATE_TOKEN}  Metric:1
          RX packets:1234 errors:0 dropped:0 overruns:0 frame:0
          TX packets:1234 errors:0 dropped:0 overruns:0 carrier:0
          collisions:0 txqueuelen:1000 
          RX bytes:{TEMPLATE_TOKEN} ({TEMPLATE_TOKEN} KB)  TX bytes:{TEMPLATE_TOKEN} ({TEMPLATE_TOKEN} KB)
"""

        messages.append({"role":"assistant", "content":template})
        return self.fill_template(messages)


    def generate_ifconfig_response(self, use_template=True):
        base_prompt = self.profile
        examples = self.get_examples("ifconfig")

        if len(examples) > 0:
            base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the ifconfig command:'
            for i in range(len(examples)):
                base_prompt = base_prompt+f"\n\nExample {i+1}:\n"+examples[i]["response"]

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"assistant", "content":""}
                ]
        messages.append({"role":"user", "content":"COMMAND: ifconfig"})

        if use_template:
            return self.generate_ifconfig_response_template(messages)
        return self.generate_from_messages(messages, max_new_tokens=1000)
#endregion

#regionnetstat
    def generate_netstat_response_template(self, messages):
        if self.num_connections is None:
            self.num_connections = np.random.randint(10, 30)
        template = f"""
        Active Internet connections (w/o servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State      
tcp        0    200 ip-172-31-22-97.eu-:ssh 138.247.230.44:53322    ESTABLISHED
Active UNIX domain sockets (w/o servers)
Proto RefCnt Flags       Type       State         I-Node   Path
unix  2      [ ]         DGRAM                    2356     /run/systemd/shutdownd
unix  3      [ ]         DGRAM                    3180     /run/systemd/notify
unix  2      [ ]         DGRAM                    3181     /run/systemd/cgroups-agent
unix  6      [ ]         DGRAM                    3187     /run/systemd/journal/socket
unix  2      [ ]         DGRAM                    2786     /run/chrony/chronyd.sock
unix  15     [ ]         DGRAM                    3188     /dev/log
unix  2      [ ]         DGRAM                    10108937 @0061b
"""

        for i in range(self.num_connections):
            connection_type = np.random.choice(["DGRAM", "STREAM"], p=[0.2, 0.8])
            state = "CONNECTED" if connection_type == "STREAM" else ""
            ref_cnt = 3 if state == "CONNECTED" else np.random.choice([2, 3], p=[0.8, 0.2])
            template += f"unix  {ref_cnt}        [ ]        {connection_type}        {state}        {TEMPLATE_TOKEN}        {TEMPLATE_TOKEN}\n"

        template += "unix  2      [ ]         DGRAM                    11188467"

        messages.append({"role":"assistant", "content":template})
        return self.fill_template(messages)


    def generate_netstat_response(self, use_template=True):
        base_prompt = self.profile
        examples = self.get_examples("netstat")

        if len(examples) > 0:
            base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the netstat command:'
            for i in range(len(examples)):
                base_prompt = base_prompt+f"\n\nExample {i+1}:\n"+examples[i]["response"]

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"assistant", "content":""}
                ]
        messages.append({"role":"user", "content":"COMMAND: netstat"})

        if use_template:
            return self.generate_netstat_response_template(messages)
        return self.generate_from_messages(messages, max_new_tokens=1000)
#endregion

#region lscpu
    def generate_lscpu_response(self):
        base_prompt = self.get_profile()
        template = f"""Architecture:          {TEMPLATE_TOKEN}
CPU op-mode(s):        {TEMPLATE_TOKEN}, {TEMPLATE_TOKEN}
Byte Order:            {TEMPLATE_TOKEN}
CPU(s):                {TEMPLATE_TOKEN}
On-line CPU(s) list:   {TEMPLATE_TOKEN}
Thread(s) per core:    {TEMPLATE_TOKEN}
Core(s) per socket:    {TEMPLATE_TOKEN}
Socket(s):             {TEMPLATE_TOKEN}
NUMA node(s):          {TEMPLATE_TOKEN}
Vendor ID:             {TEMPLATE_TOKEN}
CPU family:            {TEMPLATE_TOKEN}
Model:                 {TEMPLATE_TOKEN}
Stepping:              {TEMPLATE_TOKEN}
CPU MHz:               {TEMPLATE_TOKEN}
BogoMIPS:              {TEMPLATE_TOKEN}
Hypervisor vendor:     {TEMPLATE_TOKEN}
Virtualization type:   {TEMPLATE_TOKEN}
L1d cache:             {TEMPLATE_TOKEN}
L1i cache:             {TEMPLATE_TOKEN}
L2 cache:              {TEMPLATE_TOKEN}
NUMA node0 CPU(s):     {TEMPLATE_TOKEN}
"""
        examples = self.get_examples("lscpu")
        base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the lscpu command'

        for i in range(len(examples)):
            base_prompt = base_prompt+f"\n\nExample {i+1}:\n"+examples[i]["response"]

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"assistant", "content":""}
                ]
        messages.append({"role":"user", "content":"lscpu"})
        messages.append({"role":"assistant", "content":template})
        return self.fill_template(messages)
#endregion

#region free
    def generate_free_response(self):
        base_prompt = self.get_profile()
        template = """              total        used        free      shared  buff/cache   available
Mem:{TEMPLATE_TOKEN:>15}{TEMPLATE_TOKEN:>12}{TEMPLATE_TOKEN:>12}{TEMPLATE_TOKEN:>12}{TEMPLATE_TOKEN:>12}{TEMPLATE_TOKEN:>12}
Swap:{TEMPLATE_TOKEN:>14}{TEMPLATE_TOKEN:>12}{TEMPLATE_TOKEN:>12}
"""
        examples = self.get_examples("free")
        base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the lscpu command'

        for i in range(len(examples)):
            base_prompt = base_prompt+f"\n\nExample {i+1}:\n"+examples[i]["response"]

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"assistant", "content":""}
                ]
        messages.append({"role":"user", "content":"lscpu"})
        messages.append({"role":"assistant", "content":template})
        return self.fill_template(messages)
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
            #print(f"decoded: '{decoded}'")
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