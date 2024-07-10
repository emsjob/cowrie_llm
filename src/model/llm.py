import os
#To be added for LLM
if os.environ["COWRIE_USE_LLM"].lower() == "true":
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
    import torch
import re
import json

RESPONSE_PATH = "/cowrie/cowrie-git/src/model"
PROMPTS_PATH = "/cowrie/cowrie-git/src/model/prompts"

TEMPLATE_TOKEN = "<unk>"
TEMPLATE_TOKEN_ID = 0
SYSTEM_ROLE_AVAILABLE = True


with open(f"{RESPONSE_PATH}/cmd_lookup.json", "r") as f:
    LOOKUPS = json.load(f)

class LLM:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        with open(f"{RESPONSE_PATH}/token.txt", "r") as f:
            token = f.read().rstrip()

        self.profile = self.get_profile()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

    #def fill_template(template):
        #pass

    
    def generate_dynamic_content(self, base_prompt, dynamic_part):
        messages = [
            {"role": "user", "content": base_prompt},
            {"role": "assistant", "content": dynamic_part},
        ]
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.device)
        len_chat = tokenized_chat.shape[1]
        outputs = self.model.generate(tokenized_chat, max_new_tokens=50)
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        return response.strip()
    
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


    def fill_template(self, messages):
        tokenized_template = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")

        print("tokenized:\n", tokenized_template)
        holes = tokenized_template == TEMPLATE_TOKEN_ID
        hole_indices = holes.nonzero()

        def has_whitespace(token):
            decoded = tokenizer.decode(token)
            return decoded[0] == " "

        for hole_i in hole_indices:
            before = tokenized_template[0, :hole_i]
            after = tokenized_template[0, hole_i+1:]

            last = 1
            while not has_whitespace(last):
                before = model.generate(before, max_new_tokens=1)
                print("before:\n", before)
                last = before[0, -1]





    def generate_ifconfig_response_template(self):
        profile = self.get_profile()
        template = f"""
eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu {TEMPLATE_TOKEN}
    inet {TEMPLATE_TOKEN}  netmask {TEMPLATE_TOKEN}  broadcast {TEMPLATE_TOKEN}
    inet6 {TEMPLATE_TOKEN}  prefixlen 64  scopeid 0x20<link>
    ether {TEMPLATE_TOKEN}  txqueuelen {TEMPLATE_TOKEN}  (Ethernet)
    RX packets {TEMPLATE_TOKEN}  bytes {TEMPLATE_TOKEN} ({TEMPLATE_TOKEN})
    RX errors {TEMPLATE_TOKEN}  dropped {TEMPLATE_TOKEN}  overruns {TEMPLATE_TOKEN}  frame {TEMPLATE_TOKEN}
    TX packets {TEMPLATE_TOKEN}  bytes {TEMPLATE_TOKEN} ({TEMPLATE_TOKEN})
    TX errors {TEMPLATE_TOKEN}  dropped {TEMPLATE_TOKEN}  overruns {TEMPLATE_TOKEN}  carrier {TEMPLATE_TOKEN}  collisions {TEMPLATE_TOKEN}

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu {TEMPLATE_TOKEN}
    inet {TEMPLATE_TOKEN}  netmask {TEMPLATE_TOKEN}
    inet6 {TEMPLATE_TOKEN}  prefixlen 128  scopeid 0x10<host>
    loop  txqueuelen {TEMPLATE_TOKEN}  (Local Loopback)
    RX packets {TEMPLATE_TOKEN}  bytes {TEMPLATE_TOKEN} ({TEMPLATE_TOKEN})
    RX errors {TEMPLATE_TOKEN}  dropped {TEMPLATE_TOKEN}  overruns {TEMPLATE_TOKEN}  frame {TEMPLATE_TOKEN}
    TX packets {TEMPLATE_TOKEN}  bytes {TEMPLATE_TOKEN} ({TEMPLATE_TOKEN})
    TX errors {TEMPLATE_TOKEN}  dropped {TEMPLATE_TOKEN}  overruns {TEMPLATE_TOKEN}  carrier {TEMPLATE_TOKEN}  collisions {TEMPLATE_TOKEN}
"""
        base_prompt = profile
        examples = self.get_examples("ifconfig")

        if len(examples) > 0:
            base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the ifconfig command'

            for i in range(len(examples)):
                base_prompt = base_prompt+f"\n\nExample {i+1}\n:"+examples[i]["response"]
        print(base_prompt)

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"model", "content":""}
                ]
        messages.append({"role":"user", "content":"ifconfig"})
        messages.append({"role":"model", "content":template})
        return self.fill_template(messages)




    def generate_ifconfig_response(self, base_prompt):
        static_ifconfig_template = """eth0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu {eth0_mtu}
        inet {eth0_ip_address}  netmask {eth0_netmask}  broadcast {eth0_broadcast}
        inet6 {eth0_ipv6_address}  prefixlen 64  scopeid 0x20<link>
        ether {eth0_mac_address}  txqueuelen {eth0_txqueuelen}  (Ethernet)
        RX packets {eth0_rx_packets}  bytes {eth0_rx_bytes} ({eth0_rx_human_readable_bytes})
        RX errors {eth0_rx_errors}  dropped {eth0_rx_dropped}  overruns {eth0_rx_overruns}  frame {eth0_rx_frame}
        TX packets {eth0_tx_packets}  bytes {eth0_tx_bytes} ({eth0_tx_human_readable_bytes})
        TX errors {eth0_tx_errors}  dropped {eth0_tx_dropped}  overruns {eth0_tx_overruns}  carrier {eth0_tx_carrier}  collisions {eth0_collisions}

lo: flags=73<UP,LOOPBACK,RUNNING>  mtu {lo_mtu}
        inet {lo_ip_address}  netmask {lo_netmask}
        inet6 {lo_ipv6_address}  prefixlen 128  scopeid 0x10<host>
        loop  txqueuelen {lo_txqueuelen}  (Local Loopback)
        RX packets {lo_rx_packets}  bytes {lo_rx_bytes} ({lo_rx_human_readable_bytes})
        RX errors {lo_rx_errors}  dropped {lo_rx_dropped}  overruns {lo_rx_overruns}  frame {lo_rx_frame}
        TX packets {lo_tx_packets}  bytes {lo_tx_bytes} ({lo_tx_human_readable_bytes})
        TX errors {lo_tx_errors}  dropped {lo_tx_dropped}  overruns {lo_tx_overruns}  carrier {lo_tx_carrier}  collisions {lo_collisions}
        """

        dynamic_prompt = (
            "Generate realistic values for the following variables for an ifconfig command on a Linux terminal of a hospital server:\n"
            "eth0_ip_address, eth0_netmask, eth0_broadcast, eth0_ipv6_address, eth0_mac_address, eth0_txqueuelen, eth0_rx_packets, eth0_rx_bytes, eth0_rx_human_readable_bytes, "
            "eth0_rx_errors, eth0_rx_dropped, eth0_rx_overruns, eth0_rx_frame, eth0_tx_packets, eth0_tx_bytes, eth0_tx_human_readable_bytes, eth0_tx_errors, eth0_tx_dropped, "
            "eth0_tx_overruns, eth0_tx_carrier, eth0_collisions, eth0_mtu, lo_ip_address, lo_netmask, lo_ipv6_address, lo_txqueuelen, lo_rx_packets, lo_rx_bytes, lo_rx_human_readable_bytes, "
            "lo_rx_errors, lo_rx_dropped, lo_rx_overruns, lo_rx_frame, lo_tx_packets, lo_tx_bytes, lo_tx_human_readable_bytes, lo_tx_errors, lo_tx_dropped, lo_tx_overruns, lo_tx_carrier, lo_collisions, lo_mtu."
        )

        dynamic_content = self.generate_dynamic_content(base_prompt.format(cmd="ifconfig"), dynamic_prompt)
        dynamic_values = dict(re.findall(r"(\w+):\s*([^\n]+)", dynamic_content))

        default_values = {
            "eth0_ip_address": "192.168.1.2",
            "eth0_netmask": "255.255.255.0",
            "eth0_broadcast": "192.168.1.255",
            "eth0_ipv6_address": "fe80::21a:92ff:fe7a:672d",
            "eth0_mac_address": "00:1A:92:7A:67:2D",
            "eth0_txqueuelen": "1000",
            "eth0_rx_packets": "123456",
            "eth0_rx_bytes": "987654321",
            "eth0_rx_human_readable_bytes": "987.6 MB",
            "eth0_rx_errors": "0",
            "eth0_rx_dropped": "0",
            "eth0_rx_overruns": "0",
            "eth0_rx_frame": "0",
            "eth0_tx_packets": "123456",
            "eth0_tx_bytes": "987654321",
            "eth0_tx_human_readable_bytes": "987.6 MB",
            "eth0_tx_errors": "0",
            "eth0_tx_dropped": "0",
            "eth0_tx_overruns": "0",
            "eth0_tx_carrier": "0",
            "eth0_collisions": "0",
            "eth0_mtu": "1500",
            "lo_ip_address": "127.0.0.1",
            "lo_netmask": "255.0.0.0",
            "lo_ipv6_address": "::1/128",
            "lo_txqueuelen": "1000",
            "lo_rx_packets": "1234",
            "lo_rx_bytes": "123456",
            "lo_rx_human_readable_bytes": "123.4 KB",
            "lo_rx_errors": "0",
            "lo_rx_dropped": "0",
            "lo_rx_overruns": "0",
            "lo_rx_frame": "0",
            "lo_tx_packets": "1234",
            "lo_tx_bytes": "123456",
            "lo_tx_human_readable_bytes": "123.4 KB",
            "lo_tx_errors": "0",
            "lo_tx_dropped": "0",
            "lo_tx_overruns": "0",
            "lo_tx_carrier": "0",
            "lo_collisions": "0",
            "lo_mtu": "65536"
        }

        combined_values = {**default_values, **dynamic_values}
        ifconfig_response = static_ifconfig_template.format(**combined_values)

        return ifconfig_response

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
        """
        base_prompt = base_prompt + "\n\nHere are an example of a response to the lscpu command:"
        base_prompt = base_prompt + f"\n\n{examples['response']}"

        messages = [
            {"role": "user", "content": base_prompt}
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        len_chat = tokenized_chat.shape[1]
        outputs = self.model.generate(tokenized_chat, max_new_tokens=250)
        response = self.tokenizer.decode(outputs[0][len_chat:], skip_special_tokens=True)
        return response.strip()
        """
        base_prompt = base_prompt + f'\n\nHere {"are a few examples" if len(examples) > 1 else "is an example"} of a response to the lscpu command'

        for i in range(len(examples)):
            base_prompt = base_prompt+f"\n\nExample {i+1}\n:"+examples[i]["response"]
        print(base_prompt)

        if SYSTEM_ROLE_AVAILABLE:
            messages = [
                {"role":"system", "content":base_prompt}
                ]
        else:
            messages = [
                {"role":"user", "content":base_prompt},
                {"role":"model", "content":""}
                ]
        messages.append({"role":"user", "content":"lscpu"})
        messages.append({"role":"model", "content":template})
        return self.fill_template(messages)

class FakeLLM:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def func(*args, **kwargs):
            return "Something generated by a LLM"
        return func