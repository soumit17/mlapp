import sys
import site
site.addsitedir('/home/ubuntu/.local/lib/python3.8/site-packages')
sys.path.insert(0, '/home/ubuntu/mlapp')
from app import app as application
