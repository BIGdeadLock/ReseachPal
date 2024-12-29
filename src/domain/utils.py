import tiktoken


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base" ) -> int:
    """
    Returns the number of tokens in a text string.

    :param string: The input text string.
    :param encoding_name: The name of the encoding (e.g., 'cl100k_base').
    :return: The number of tokens in the string.
    """
    # Load the specified encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Encode the string and count tokens
    num_tokens = len(encoding.encode(string))

    return num_tokens