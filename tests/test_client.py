
import asyncio
import websockets
import json

async def test_client():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        
        # Send a message
        message = {"type": "input", "content": "Hello"}
        await websocket.send(json.dumps(message))
        print(f"Sent: {message}")
        
        # Listen for responses
        for _ in range(10):
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                data = json.loads(response)
                print(f"Received: {data}")
                if data['type'] == 'status' and data['content'] == 'WAIT':
                    print("Agent is waiting. Test passed.")
                    break
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
                break
                
if __name__ == "__main__":
    asyncio.run(test_client())
