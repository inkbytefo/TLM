
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import json
import logging

from autonomous_agent import AgentLoop, SILENCE_TOKEN, WAIT_TOKEN, THINK_TOKEN, SPEAK_TOKEN

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize Agent
agent = AgentLoop()

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        # Background task to run the agent loop
        # In a real full-duplex system, we need to handle incoming messages AND run the agent loop concurrently.
        
        # We'll use a queue to pass user input to the agent loop
        input_queue = asyncio.Queue()
        
        async def receive_messages():
            """Listen for messages from client and put them in queue."""
            try:
                while True:
                    data = await websocket.receive_text()
                    # data is a JSON string: {"type": "input", "content": "..."}
                    message = json.loads(data)
                    if message["type"] == "input":
                        content = message["content"]
                        # Push each character/token to queue
                        for char in content:
                            await input_queue.put(ord(char))
                        # Push newline
                        await input_queue.put(ord('\n'))
            except Exception as e:
                logger.error(f"Receive error: {e}")

        async def agent_loop():
            """Run the agent step loop."""
            silence_counter = 0
            
            while True:
                # 1. Check for Input
                if not input_queue.empty():
                    # Process all available input
                    while not input_queue.empty():
                        token = await input_queue.get()
                        # Feed to agent (update memory)
                        _ = agent.step(token)
                        # Echo back to client? No, client shows what user typed.
                        # But we might want to acknowledge receipt?
                    
                    # After processing input, we can optionally trigger an immediate response
                    # or just let the loop continue.
                    last_output = agent.step(SILENCE_TOKEN) # Trigger next step
                    
                else:
                    # No input, feed Silence
                    last_output = agent.step(SILENCE_TOKEN)
                    silence_counter += 1
                
                # 2. Handle Agent Output
                current_token = last_output
                
                # If agent wants to speak
                if current_token == SPEAK_TOKEN:
                    await websocket.send_json({"type": "status", "content": "SPEAK"})
                    
                    # Generate speech until WAIT or SILENCE
                    # Limit generation to avoid blocking too long
                    for _ in range(50):
                        # Generate next
                        current_token = agent.step(current_token)
                        
                        if current_token == WAIT_TOKEN:
                            await websocket.send_json({"type": "status", "content": "WAIT"})
                            break
                        elif current_token == SILENCE_TOKEN:
                            await websocket.send_json({"type": "status", "content": "SILENCE"})
                            break
                        elif current_token == THINK_TOKEN:
                            await websocket.send_json({"type": "status", "content": "THINK"})
                            # Continue loop but don't send text
                        elif current_token == SPEAK_TOKEN:
                            pass # Already speaking
                        else:
                            # Text token
                            try:
                                char = bytes([current_token]).decode('utf-8')
                                await websocket.send_json({"type": "token", "content": char})
                            except:
                                pass
                        
                        # Small delay to simulate typing speed / allow interrupt check
                        await asyncio.sleep(0.02)
                        
                        # Check for interruption (optional optimization)
                        if not input_queue.empty():
                            # User interrupted!
                            # We could break here to process input immediately
                            break
                            
                elif current_token == THINK_TOKEN:
                    await websocket.send_json({"type": "status", "content": "THINK"})
                    
                elif current_token == WAIT_TOKEN:
                    # Agent is waiting
                    # Don't spam status updates
                    pass
                    
                # Control loop speed
                await asyncio.sleep(0.05) # 20Hz loop

        # Run both tasks
        receiver_task = asyncio.create_task(receive_messages())
        agent_task = asyncio.create_task(agent_loop())
        
        # Wait for either to finish (likely receiver on disconnect)
        done, pending = await asyncio.wait(
            [receiver_task, agent_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
