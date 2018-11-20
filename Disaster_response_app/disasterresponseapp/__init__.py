from flask import Flask

app = Flask(__name__)
from files.essential import tokenize
from disasterresponseapp import run
