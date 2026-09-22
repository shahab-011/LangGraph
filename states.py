# so now we are creating a graph 
# and the first thing i create is state 

import os

# 1. Typed Dictionary
from typing import TypedDict
class State(TypedDict):
    topic : str
    summary : str
    score : str

# 2. Pydantic Approach 
# it si good at data validation and type chaecing at run time 

 