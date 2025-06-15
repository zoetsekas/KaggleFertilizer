import argparse
from enum import Enum, EnumMeta

parser = argparse.ArgumentParser()

class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

class Action(str, Enum, metaclass=MetaEnum):
    TRAIN = 'train'
    INFERENCE = 'inference'

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    parser.add_argument('--action', type=Action, required=True, default='start')

    args = parser.parse_args()

    action = args.action

    if action == Action.TRAIN:
        pass
    elif action == Action.INFERENCE:
        pass
    else:
        print("Invalid action")
