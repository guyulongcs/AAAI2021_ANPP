import datetime
import math

class Tool():
    @classmethod
    def output(cls, name, variable, flagDebug=True):
        if(not flagDebug):
            return
        print("\n%s:" % name)
        print(variable)

    @classmethod
    def write_list_list_to_file(cls, listlist, file, sep='\t'):
        with open(file, 'w') as f:
            for list in listlist:
                list = [str(item) for item in list]
                line = sep.join(list)
                f.write(line + "\n")
