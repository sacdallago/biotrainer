def is_running_in_notebook() -> bool:
    try:
        # This will only exist in Jupyter environments
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':  # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':  # IPython terminal
            return False
    except NameError:
        return False
    return False