import sys
sys.path.append("/cowrie/cowrie-git/src")

from model.llm import LLM, FakeLLM
import hashlib
import json
import os

TEXTCMDS_PATH = "/cowrie/cowrie-git/share/cowrie/txtcmds"

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


#build for lscpu

try:
    lscpu_resp = static_cache[profile_hash]["lscpu"]
except KeyError:
    if llm is None:
        llm = LLM()
    lscpu_resp = llm.generate_lscpu_response()

#raise SystemExit(0)

if llm is None:
    llm = LLM()

lscpu_resp = llm.generate_lscpu_response()
if lscpu_resp[-1] != "\n":
    lscpu_resp += "\n"
LSCPU_PATH = TEXTCMDS_PATH+"/usr/bin/lscpu"

with open(LSCPU_PATH, "r") as lscpu_file:
    print("CPU BEFORE: ", lscpu_file.read())

with open(LSCPU_PATH, "w") as lscpu_file:
    lscpu_file.write(lscpu_resp)

with open(LSCPU_PATH, "r") as lscpu_file:
    print("CPU AFTER: ", lscpu_file.read())

nproc_resp = llm.generate_nproc_response()
if nproc_resp[-1] != "\n":
    nproc_resp += "\n"
NPROC_PATH = TEXTCMDS_PATH+"/usr/bin/nproc"

with open(NPROC_PATH, "r") as nproc_file:
    print("NPROC BEFORE: ", nproc_file.read())

with open(NPROC_PATH, "w") as nproc_file:
    nproc_file.write(nproc_resp)

with open(NPROC_PATH, "r") as nproc_file:
    print("NPROC AFTER: ", nproc_file.read())

df_resp = llm.generate_df_response()
if df_resp[-1] != "\n":
    df_resp += "\n"
DF_PATH = TEXTCMDS_PATH+"/bin/df"

with open(DF_PATH, "r") as df_file:
    print("DF BEFORE: ", df_file.read())

with open(DF_PATH, "w") as df_file:
    df_file.write(df_file)

with open(DF_PATH, "r") as df_file:
    print("DF AFTER: ", df_file.read())




