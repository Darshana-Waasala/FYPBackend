import asyncio
import datetime
import json
import random
import websockets


STATE = {'value': 0}


def state_event():
    return json.dumps({'type': 'state', **STATE})


async def echo(websocket, path):
    print('started web socket.')
    async for message in websocket:
        print('got a message form client')
        await websocket.send(message)


async def time(websocket, path):
    while True:
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        await websocket.send(now)
        await asyncio.sleep(random.random() * 3)

start_server = websockets.serve(echo, '127.0.0.1', 5678)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()