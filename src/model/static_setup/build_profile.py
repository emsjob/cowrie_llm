import sys
sys.path.append("/cowrie/cowrie-git/src")
from model.llm import LLM, FakeLLM
import hashlib
import json
import os
from os import path
from cowrie.core.config import CowrieConfig
from cowrie.scripts import fsctl
import re

TEXTCMDS_PATH = "/cowrie/cowrie-git/share/cowrie/txtcmds"
HONEYFS_PATH = "/cowrie/cowrie-git/honeyfs"

CACHE_PATH = "/cowrie/cowrie-git/src/model/static_setup/static_cache.json"
with open(CACHE_PATH) as cache_file:
    static_cache = json.load(cache_file)

PROFILE_PATH = "/cowrie/cowrie-git/src/model/prompts/large_profile.txt"
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
for _ in range(3):
    try:
        lscpu_resp = get_resp("lscpu", "generate_lscpu_response")
        break
    except Exception as error:
        print("lscpu response failed, retrying...")
        print("Error: ", error)
if lscpu_resp[-1] != "\n":
    lscpu_resp += "\n"

LSCPU_PATH = TEXTCMDS_PATH+"/usr/bin/lscpu"

with open(LSCPU_PATH, "w") as lscpu_file:
    lscpu_file.write(lscpu_resp)
#endregion

#region nproc
cpu_count = 0
for line in lscpu_resp.splitlines():
    if line.startswith("CPU(s):"):
        cpu_count = line.split(":")[1].strip()
        break

nproc_resp = cpu_count
if nproc_resp[-1] != "\n":
    nproc_resp += "\n"
NPROC_PATH = TEXTCMDS_PATH+"/usr/bin/nproc"

with open(NPROC_PATH, "w") as nproc_file:
    nproc_file.write(nproc_resp)
#endregion

#region df
for _ in range(3):
    try:
        df_resp = get_resp("df", "generate_df_response")
        break
    except Exception as error:
        print("df response failed, retrying...")
        print("Error: ", error)
if df_resp[-1] != "\n":
    df_resp += "\n"
DF_PATH = TEXTCMDS_PATH+"/bin/df"

with open(DF_PATH, "w") as df_file:
    df_file.write(df_resp)
#endregion

#region hostname
for _ in range(3):
    try:
        hostname_resp = get_resp("hostname", "generate_host_name")
        break
    except Exception as error:
        print("hostname response failed, retrying...")
        print("Error: ", error)
if hostname_resp[-1] != "\n":
    hostname_resp = hostname_resp+"\n"
print("Generated hostname:", hostname_resp)

CowrieConfig.set("honeypot", "hostname", hostname_resp)
#endregion

#region users
with open("/cowrie/cowrie-git/honeyfs/etc/passwd", "r") as passwd_file:
    base_passwd = passwd_file.read()

for _ in range(3):
    try:
        new_passwd = llm.add_users(base_passwd)
        break
    except Exception as error:
        print("adding users failed, retrying...")
        print("Error: ", error)
print(f"new passwd:\n{new_passwd}")

if new_passwd[-1] != "\n":
    new_passwd += "\n"

with open("/cowrie/cowrie-git/honeyfs/etc/passwd", "w") as passwd_file:
    passwd_file.write(new_passwd)

users = [line.split(":")[0] for line in new_passwd.split("\n")]
#This relies on the admin user existing at the end of the base userdb.txt file
new_i = users.index("admin")
users = users[new_i:]
print("new users:", users)

fscmd = fsctl.fseditCmd("/cowrie/cowrie-git/share/cowrie/fs.pickle")
fscmd.pickle_file_path = "/cowrie/cowrie-git/share/cowrie/fs2.pickle"

fscmd.do_rm("-r /home")
fscmd.do_mkdir("/home false")
with open("/cowrie/cowrie-git/etc/userdb.txt", "a") as userdb_file:
    print("starting loop")
    for user in users:
        if user:
            user = user.split("/")[-1]
            print("adding user:", user)
            fscmd.do_mkdir(f"/home/{user} llm")
            userdb_file.write(f"{user}:x:/{user}/i\n")

#Unnecessary?
fscmd.save_pickle()

CowrieConfig.set("shell", "filesystem", "${honeypot:share_path}/fs2.pickle")
#endregion

#Save changes to config file
#This might duplicate the config settings into one file, since original CowrieConfig is loaded from multiple
#Potential bug, likely harmless if it works
with open("/cowrie/cowrie-git/etc/cowrie.cfg", "w") as configfile:
    CowrieConfig.write(configfile)