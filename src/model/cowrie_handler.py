import random
import time
import os
import re
import stat

class CowrieHandler():
    def __init__(self, protocol) -> None:
        self.protocol = protocol
        self.fs = protocol.fs
        self.enforced_ls_paths = []
    
    def enforce_ls(self, path: str, ls_view: str):
        if not self.fs.exists(path):
            return
        if path in self.enforced_ls_paths:
            return

        items = re.split(r'\s+', ls_view)
        print("ls items:", items)
        def is_file(item: str):
            return "." in item
        
        def random_time(months_ago):
            ctime = time.time()
            return ctime-random.uniform(0, months_ago*30*24*60*60)

        def random_size():
            return random.randrange(1024, int(1e6), 1024)
        
        perm = stat.S_IFREG |stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
        for item in items:
            "make sure item is a item and not a path, if path select last"
            item = item.split("/")[-1]

            item_path = path+"/"+item
            if not self.fs.exists(item_path):
                if is_file(item):
                    self.fs.mkfile(item_path, 0, 0, random_size(), perm, random_time(6), is_llm=True)
                else:
                    self.fs.mkdir(item_path, 0, 0, 4096, perm, random_time(6), is_llm=True)
        self.enforced_ls_paths.append(path)

    def enforce_file_contents(self, path, contents):
        tmp_fname = "{}-{}-{}-redir_{}".format(
            time.strftime("%Y%m%d-%H%M%S"),
            self.protocol.getProtoTransport().transportId,
            self.protocol.terminal.transport.session.id,
            re.sub("[^A-Za-z0-9]", "_", path),
        )
        #TODO: Change this to get from CowrieConfig
        safeoutfile = "var/lib/cowrie/downloads/"+tmp_fname

        data = bytes(contents, encoding='utf-8')
        with open(safeoutfile, "ab") as f:
            f.write(data)

        fs_f = self.fs.getfile(path)
        self.fs.update_realfile(fs_f, safeoutfile)
        self.fs.update_file_size(fs_f, len(data))




