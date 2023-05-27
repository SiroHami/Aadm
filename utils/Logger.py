import sys

class Logger(object): #you should logger=Logger() and logger.open("asdasd.txt")
    def __init__(self, local_rank=0, no_save=False):
        self.terminal = sys.stdout #like print, sysout to disply, faster than print
        self.file = None #open notepad(.txt) and save it
        self.local_rank = local_rank
        self.no_save = no_save
    def open(self, fp, mode=None):
        if mode is None: mode = 'w'
        if self.local_rank and not self.no_save == 0: self.file = open(fp, mode)
    def write(self, msg, is_terminal=1, is_file=1):
        if msg[-1] != "\n": msg = msg + "\n"
        if self.local_rank == 0:
            if '\r' in msg: is_file = 0
            if is_terminal == 1:
                self.terminal.write(msg)
                self.terminal.flush()
            if is_file == 1 and not self.no_save:
                self.file.write(msg)
                self.file.flush()
    def flush(self): 
        pass