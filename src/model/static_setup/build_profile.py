import sys
sys.path.append("/cowrie/cowrie-git/src")

from model.llm import LLM, FakeLLM
import hashlib
import json
import os
from cowrie.core.config import CowrieConfig

TEXTCMDS_PATH = "/cowrie/cowrie-git/share/cowrie/txtcmds"
HONEYFS_PATH = "/cowrie/cowrie-git/honeyfs"

CACHE_PATH = "/cowrie/cowrie-git/src/model/static_setup/static_cache.json"
with open(CACHE_PATH) as cache_file:
    static_cache = json.load(cache_file)

PROFILE_PATH = "/cowrie/cowrie-git/src/model/prompts/profile.txt"
with open(PROFILE_PATH) as profile_file:
    profile = profile_file.read()
profile_bare = "".join(filter(str.isalpha, profile.lower()))
profile_hash = hashlib.sha256(profile_bare.encode("utf-8")).hexdigest()

#If not using LLM set to fake one, otherwise leave none and instantiate real one if necessary
if os.environ["COWRIE_USE_LLM"].lower() == "true":
    llm = None
else:
    llm = FakeLLM()

def get_resp(name, func_name):
    global llm
    try:
        resp = static_cache[profile_hash][name]
    except KeyError:
        if llm is None:
            llm = LLM()
        resp = llm.__getattribute__(func_name)()
    return resp
    
#region lscpu
lscpu_resp = get_resp("lscpu", "generate_lscpu_response")

LSCPU_PATH = TEXTCMDS_PATH+"/usr/bin/lscpu"

with open(LSCPU_PATH, "w") as lscpu_file:
    lscpu_file.write(lscpu_resp)
#endregion

#region hostname
hostname_resp = get_resp("hostname", "generate_host_name")
if hostname_resp[-1] != "\n":
    hostname_resp = hostname_resp+"\n"
print("Generated hostname:", hostname_resp)

CowrieConfig.set("honeypot", "hostname", hostname_resp)
#endregion


#Save changes to config file
#This might duplicate the config settings into one file, since original CowrieConfig is loaded from multiple
#Potential bug, likely harmless if it works
with open("/cowrie/cowrie-git/etc/cowrie.cfg", "w") as configfile:
    CowrieConfig.write(configfile)

