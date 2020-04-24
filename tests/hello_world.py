from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
import os

@task(returns=1)
def create_greeting(message, use_storage):
    """
    Instantiates a persistent object and populates it with the received
    message.
    :param message: String with the information to store in the psco.
    :return: The populated persistent object.
    """
    if use_storage:
        from storage_model.classes import hello
    else:
        from model.classes import hello
    print("vaaaarsworker")
    print(os.environ)
    if use_storage:
        hi = hello("greet")
        hi.message = message
        #hi.make_persistent()
    else:
        hi = hello()
        hi.message = message
    return hi


@task(returns=1)
def greet(greetings):
    """
    Retrieves the information contained in the given persistent object.
    :param greetings: Persistent object.
    :return: String with the psco content.
    """
    content = greetings.message
    return content


@task(returns=1)
def check_greeting(content, message):
    """
    Checcks that the given content is equal to the given message.
    :param content: String with content.
    :param message: String with message.
    :return: Boolean (True if equal, False otherwise).
    """
    return content == message


def parse_arguments():
    """
    Parse command line arguments. Make the program generate
    a help message in case of wrong usage.
    :return: Parsed arguments
    """
    import argparse
    parser = argparse.ArgumentParser(description='Hello world.')
    parser.add_argument('--use_storage', action='store_true',
                        help='Use storage?')
    return parser.parse_args()


def main(use_storage):
    # import sys
    # sys.path.append("./debug/pydevd-pycharm.egg")
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('192.168.1.222', port=12345, stdoutToServer=True, stderrToServer=True)
    print("vaaaars")
    print(os.environ)
    message = "Hello world"
    greeting = create_greeting(message, use_storage)
    content = greet(greeting)
    result = check_greeting(content, message)
    result_wrong = check_greeting(content, message + "!!!")
    result = compss_wait_on(result)
    result_wrong = compss_wait_on(result_wrong)
    if result != result_wrong:
        print("THE RESULT IS OK")
    else:
        msg = "SOMETHING FAILED!!!"
        print(msg)
        raise Exception(msg)


if __name__ == "__main__":
    options = parse_arguments()
    main(**vars(options))