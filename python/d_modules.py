
import sys

a = 5
zzzzzzzzzzzzz = 'sample'

if __name__ == '__main__':
    print('this program is being run by itself')
else:
    print('i am being imported by another module')

# ########################
# this uses the dir() function
# dir() will print all the methods and attributes of the current class or a module
def print_all_system_items():
    for item in dir(sys):
        print(item)

def get_all_d_module_info():
    """this will give all the information about the module"""
    a_local_var = 'this is local variable'
    zzz = 5


__version__ = 0.01
