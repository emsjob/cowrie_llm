import json
from model.cowrie_handler import CowrieHandler
from model.llm import LLM, FakeLLM, ServiceLLM
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
            self.response_dict = json.load(response_file)

        self.history = []

    def record_response(self, cmd, response, **kwargs):
        message = {"cmd":cmd,
                   "response":response}
        for key, value in kwargs.items():
            message[key] = value
        self.history.append(message)
        return
    
    def get_cmds_history(self, cmds):
        print("history:\n", self.history)
        return [entry for entry in self.history if entry["cmd"] in cmds]
            
    
    def ls_respond(self,
                   path: str):
        resp = self.find_static_response("ls", "", path)

        if resp is None:
            ls_history = self.get_cmds_history(["ls"])
            print("ls_history:\n", ls_history)
            resp = self.llm.generate_ls_response(path, history=ls_history)
        
        #Should maybe be just for new LLM generations?
        print("RESPONSE!!")
        print(resp)
        print("------")
        self.ch.enforce_ls(path, resp)
        self.record_response("ls", resp, path=path)

    def netstat_respond(self):
        #resp = self.find_static_response("netstat")
        #if resp is None:
        resp = self.llm.generate_netstat_response()
        print("RESPONSE!!")
        print(resp)
        print("------")

        return resp
        
    def ifconfig_respond(self):
        #resp = self.find_static_response("ifconfig")
        #if resp is None:
        resp = self.llm.generate_ifconfig_response()
        print("ifconfig RESPONSE!!")
        print(resp)
        print("------")
        return resp

    def free_respond(self):
        resp = self.llm.generate_free_response()
        return resp

    def lscpu_respond(self):
        resp = self.llm.generate_lscpu_response()
        return resp

    def last_respond(self):
        resp = self.llm.generate_last_response()
        return resp

    def find_static_response(self,
                      command:str,
                      flags: list[str] = "",
                      path: None | str = None
                      ):
        if path is not None:
            try:
                return self.response_dict[command][flags][path]
            except Exception:
                return None
        else:
            try:
                return self.response_dict[command][flags]
            except Exception:
                return None
            
    def file_contents_respond(self, path: str):
        resp = self.find_static_response("file_contents", "", path)
        if resp is None:
            resp = self.llm.generate_file_contents(path)
        self.ch.enforce_file_contents(path, resp)
        return resp
