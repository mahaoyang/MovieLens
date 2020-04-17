from redis import Redis

rc = Redis()

rc.set('123', 0)
a = rc.get('123').decode()
print(type(a))
