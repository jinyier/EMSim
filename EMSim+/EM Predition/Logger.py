import sys  # 需要引入的包


# 以下为包装好的 Logger 类的定义
class Logger(object):
    def __init__(self, filename="Default.log",stream = sys.stdout):
        self.terminal = stream
        self.log = open(filename, "a")
        # self.log = open(filename, "a", encoding="utf-8")  # 防止编码错误

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
