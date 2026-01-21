import sys
from logger import logging

def error_message_detail(error):
    """
    Returns detailed error message including file name and line number.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_lineno
    return f"Error occurred in Python script [{file_name}] line number [{line_no}]: {str(error)}"

class CustomException(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error_message = error_message_detail(error)
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message