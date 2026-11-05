 
import jwt as pyjwt, datetime
SECRET = 'dermai-super-secret-change-in-prod'
payload = {
    'sub': 1,
    'email': 'test@test.com',
    'name': 'test',
    'iat': datetime.datetime.utcnow(),
    'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
}
token = pyjwt.encode(payload, SECRET, algorithm='HS256')
print('TOKEN:', token)
decoded = pyjwt.decode(token, SECRET, algorithms=['HS256'])
print('DECODED:', decoded)