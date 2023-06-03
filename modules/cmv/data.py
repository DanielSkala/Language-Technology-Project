import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple, Dict, Union, Any, Optional
from tqdm import tqdm
import json

def read_data(path: str) -> List[dict]:
    """
    Read the data from the jsonl file
    Args:
        path: path to the jsonl file
    Returns:
        data: list of dictionaries
    """
    # parse the content of the json file
    text = open(path, 'r').read()
    # parse jsonl
    data = [json.loads(jline) for jline in text.split('\n') if jline]
    return data

def write_data(data: List[dict], path: str) -> None:
    """
    Write the data to the jsonl file
    Args:
        data: list of dictionaries
        path: path to the jsonl file
    Returns:
        None
    """
    with open(path, 'w') as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write('\n')

def format_data(data):
    """
    Format the data into a list of dictionaries
    Args:
        data: list of dictionaries
    Returns:
        result: list of dictionaries
    """
    result = []
    for entry in data:
        c = {
            "head": entry["title"],
            "comments": []
        }
        if len(entry["comments"]) > 0:
            stack = [entry["comments"][0]]
            while stack:
                comment = stack.pop()
                for reply in comment["children"]:
                    stack.append(reply)
                c["comments"].append(comment["body"])
        result.append(c)
    return result

def show_data(data):
    """
    Show the data
    Args:
        data: list of dictionaries
    Returns:
        None
    """
    # Should truncate the data to 10 and put ... in between
    string = ""
    for i in range(min(10, len(data))):
        string += "Head: " + data[i]["head"]
        
        for j in range(min(3, len(data[i]["comments"]))):
            string += f"\n Comment {j}: " + data[i]["comments"][j][:50] + "..."
        
        string += "\n ...\n" if len(data[i]["comments"]) > 3 else "\n"
        string += "\n"
    string += "..." if len(data) > 10 else ""
    print(string)