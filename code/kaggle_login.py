import sys
import json

login_data = {}
login_data['UserName'] = sys.argv[1]
login_data['Password'] = sys.argv[2]

with open('login_data.json', 'w') as fp:
    json.dump(login_data, fp)
