import os.path
import sys
import shutil
import requests
import json

if not os.path.isfile('login_data.json'):
	print "login_data.json not detected, Please run kaggle_login.py"

login_url = 'https://www.kaggle.com/account/login'
download_url = sys.argv[1]
filename = download_url.split('/')[-1]

def print_download():
	global filename
	print "Downloading {} in process".format(filename)


with open('login_data.json') as data_file:    
    login_data = json.load(data_file)

print_download()

with requests.session() as s, open(filename, 'w') as f:
    s.post(login_url, data=login_data)                  # login
    response = s.get(download_url, stream=True)         # send download request
    shutil.copyfileobj(response.raw, f)                 # save response to file