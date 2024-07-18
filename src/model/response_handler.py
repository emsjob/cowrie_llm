import json
from model.cowrie_handler import CowrieHandler
from model.llm import LLM, FakeLLM, ServiceLLM, SingletonMeta
import os

RESPONSE_PATH = "/cowrie/cowrie-git/src/model/static_responses.json"

class ResponseHandler():
    def __init__(self, protocol) -> None:
        protocol.fs.rh = self
        self.ch = CowrieHandler(protocol)
        if os.environ["COWRIE_USE_LLM"].lower() == "true":
            print("using llm service")
            self.llm = ServiceLLM()
        else:
            print("using fake llm")
            self.llm = FakeLLM()

        with open(RESPONSE_PATH) as response_file:
            self.static_responses = json.load(response_file)
            
        self.history = []
        self.gm = GeneralManager()

    def find_static_response(self,
                      command:str,
                      flags: list[str] = "",
                      path: str = ""
                      ):
        if path:
            try:
                return self.response_dict[command][flags][path]
            except Exception:
                return None
        else:
            try:
                return self.response_dict[command][flags]
            except Exception:
                return None
            
    def find_response(self, command, flags="", path=""):
        general_resp = self.gm.find_general_response(command, flags, path)
        if general_resp:
            return general_resp
        return self.find_static_response(command, flags, path)


    def record_response_history(self, cmd, response, **kwargs):
        message = {"cmd":cmd,
                   "response":response}
        for key, value in kwargs.items():
            message[key] = value
        self.history.append(message)
        return
    
    def record_response_gm(self, cmd, resp, flags="", path=""):
        if path:
            print("before record1:", self.gm.response_cache[cmd][flags][path])
            self.gm.response_cache[cmd][flags][path] = resp
            print("after record1:", self.gm.response_cache[cmd][flags][path])
        else:
            self.gm.response_cache[cmd][flags] = resp

    
    def get_cmds_history(self, cmds):
        print("history:\n", self.history)
        return [entry for entry in self.history if entry["cmd"] in cmds]
            
    
    def ls_respond(self,
                   path: str):
        resp = self.find_response("ls", "", path)

        if not resp:
            ls_history = self.get_cmds_history(["ls"])
            resp = self.llm.generate_ls_response(path, history=ls_history)
            print("record before:", self.gm.response_cache)
            self.record_response_gm("ls", resp, "", path)
            print("record after:", self.gm.response_cache)
        
        #Should maybe be just for new LLM generations?
        print("RESPONSE!!")
        print(resp)
        print("------")
        self.ch.enforce_ls(path, resp)
        self.record_response_history("ls", resp, path=path)
        

    def netstat_respond(self):
        resp = self.find_response("netstat")
        if not resp:
            resp = self.llm.generate_netstat_response()
            self.record_response_gm("netstat", resp)
        print("RESPONSE!!")
        print(resp)
        print("------")

        return resp
        
    def ifconfig_respond(self):
        resp = self.find_response("ifconfig")
        if not resp:
            resp = self.llm.generate_ifconfig_response()
            self.record_response_gm("ifconfig", resp)

        print("ifconfig RESPONSE!!")
        print(resp)
        print("------")
        return resp

    def free_respond(self):
        resp = self.llm.generate_free_response()
        return resp

    def lscpu_respond(self):
        resp = self.find_response("lscpu")
        if not resp:
            resp = self.llm.generate_lscpu_response()
            self.record_response_gm("lscpu", resp)
        return resp

    def last_respond(self):
        resp = self.llm.generate_last_response()
        return resp
            
    def file_contents_respond(self, path: str):
        resp = self.find_response("file_contents", "", path)
        if not resp:
            resp = self.llm.generate_file_contents(path)
            self.record_response_gm("file_contents", resp, "", path)
        self.ch.enforce_file_contents(path, resp)
        return resp



from collections import defaultdict

def def_value():
    return defaultdict(def_value)

class GeneralManager(metaclass=SingletonMeta):
    def __init__(self):
        self.response_cache = defaultdict(def_value)

    def find_general_response(self, cmd, flags="", path=""):
        print("Searching for", cmd, flags, path)
        print("in:")
        print(dict(self.response_cache))
        if not path:
            return self.response_cache[cmd][flags]
        else:
            res = self.response_cache[cmd][flags][path]
            print("found response:", res)
            return res

    