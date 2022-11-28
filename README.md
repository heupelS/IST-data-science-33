# IST-data-science-33

Final-Report: [Link](https://www.overleaf.com/1932411886vvfcqbrbrcfq)

## Setup 

From lecture its recommend to use python3.10!

Run ```pip install -r requirements.txt``` to install dependencies

## Ressources

- Website with samples: [Link](http://web.ist.utl.pt/~claudia.antunes/DSLabs/)

## Datasets 

### SET 1


### SET 2



## Don't rewrite fucntions, use modules

For using modules, which can be imported in the task, please use the ```src/utils``` folder.

Then we can include these files by the following commands on top of the python file: 

```python
import sys, os

# Add  relative path to the directory
sys.path.append( os.path.join(os.path.dirname(__file__), '..', '..','utils') )
from <Your_Library> import <Your_Function>
```