# MCP Client Guide

## Issue: "Received a response for an unknown message ID"

This error occurs when the MCP client (Cursor IDE) receives a response with a message ID it doesn't recognize. This can happen due to:

1. **Client reconnection**: Client restarts and receives stale responses
2. **Race conditions**: Multiple requests sent before previous responses arrive
3. **Message ID tracking issues**: Client loses track of pending requests

## Server-Side Robustness 

The MCP server implements the following robustness features:

### 1. Request Tracking & Deduplication

The server tracks all incoming requests by message ID:
- **Duplicate Detection**: If the same message ID is received twice, the server returns an error instead of processing it again
- **Request Cleanup**: Stale requests (>5 minutes) are automatically cleaned up
- **Initialize Reset**: When `initialize` is called, all pending requests are cleared (handles client restarts)

### 2. Enhanced Error Handling

- **Message Validation**: All messages are validated before processing
- **JSON-RPC Compliance**: Ensures all responses follow JSON-RPC 2.0 spec
- **Graceful Degradation**: Errors are returned with proper error codes instead of crashing

### 3. Debug Mode

Enable debug logging by setting:
```bash
export AISP_MCP_DEBUG=1
```

This will log:
- Duplicate request detection
- Stale request cleanup
- Request tracking information

## Robust Client Implementation

The production-ready MCP client implementation (`mcp/mcp_client.py`) handles all the robustness concerns automatically.

### Using the Robust Client

```python
from mcp.mcp_client import create_client

# Simple usage
client = create_client(debug=True)
client.start()

# List tools
tools = client.list_tools()

# Call a tool
result = client.call_tool("status", {})

# Optimize shortcut (path or target)
opt_result = client.call_tool("optimize", {"target": "ch10:atomic_reduction"})

# Stop client
client.stop()
```

### Context Manager Usage

```python
# Automatic cleanup
with create_client() as client:
    result = client.call_tool("gpu_info", {})
    # Client automatically stops on exit
```

### Features

The robust client includes:
- ✅ **Automatic message ID management** - Thread-safe, unique IDs
- ✅ **Request/response correlation** - Tracks all pending requests
- ✅ **Timeout handling** - Automatic cleanup of stale requests
- ✅ **Duplicate detection** - Handles unknown message IDs gracefully
- ✅ **Error recovery** - Proper error handling and reporting
- ✅ **Reconnection support** - Can restart server process
- ✅ **Thread-safe** - Safe for concurrent use

### Configuration

```python
client = RobustMCPClient(
    command=["python", "-m", "mcp.mcp_server", "--serve"],
    cwd="/path/to/workspace",
    timeout=300.0,  # Request timeout in seconds
    enable_debug=True  # Enable debug logging
)
```

## Client-Side Best Practices (Custom Implementations)

If you're building your own client, here are best practices:

### 1. Message ID Management

```python
class RobustMCPClient:
    def __init__(self):
        self._pending_requests = {}
        self._request_lock = threading.Lock()
        self._next_id = 1
    
    def _get_next_id(self):
        """Generate unique message IDs."""
        with self._request_lock:
            msg_id = self._next_id
            self._next_id += 1
            return msg_id
    
    def send_request(self, method, params):
        """Send request with proper tracking."""
        msg_id = self._get_next_id()
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }
        
        # Track pending request
        with self._request_lock:
            self._pending_requests[msg_id] = {
                "method": method,
                "sent_at": time.time(),
                "response": None
            }
        
        # Send request...
        return msg_id
    
    def handle_response(self, response):
        """Handle response with validation."""
        msg_id = response.get("id")
        
        if msg_id is None:
            # Notification, no response needed
            return
        
        with self._request_lock:
            if msg_id not in self._pending_requests:
                # Unknown message ID - log but don't crash
                print(f"Warning: Received response for unknown message ID: {msg_id}")
                return
            
            # Mark as received
            self._pending_requests[msg_id]["response"] = response
            # Clean up after processing
            del self._pending_requests[msg_id]
```

### 2. Timeout Handling

```python
def cleanup_stale_requests(self, timeout_seconds=300):
    """Remove requests that haven't received responses."""
    current_time = time.time()
    with self._request_lock:
        stale = [
            msg_id for msg_id, req in self._pending_requests.items()
            if current_time - req["sent_at"] > timeout_seconds
        ]
        for msg_id in stale:
            print(f"Warning: Request {msg_id} timed out")
            del self._pending_requests[msg_id]
```

### 3. Reconnection Handling

```python
def reconnect(self):
    """Handle client reconnection."""
    # Clear all pending requests
    with self._request_lock:
        self._pending_requests.clear()
    
    # Re-initialize connection
    self.send_request("initialize", {})
```

## Troubleshooting

### Enable Debug Mode

```bash
export AISP_MCP_DEBUG=1
python -m mcp.mcp_server --serve
```

### Check Server Logs

The server logs to stderr. Check for:
- Duplicate request warnings
- Stale request cleanup messages
- Error details

### Common Issues

1. **"Unknown message ID" warnings**: Usually harmless if tools still work. The server now handles duplicates gracefully.

2. **Stale responses**: Server automatically cleans up requests older than 5 minutes.

3. **Client reconnection**: Server clears pending requests on `initialize` call.

## Server Configuration

### Environment Variables

- `AISP_MCP_DEBUG=1`: Enable debug logging
- `AISP_MCP_REQUEST_TIMEOUT=300`: Request timeout in seconds (default: 300)

### Request Timeout

The server automatically cleans up requests older than 5 minutes. This prevents memory leaks from lost requests.

## Testing

Test the robustness features:

```python
# Test duplicate detection
import asyncio
from mcp.mcp_server import MCPServer

async def test_duplicate():
    server = MCPServer()
    
    # Send same request twice
    msg1 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    msg2 = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    
    resp1 = await server.handle_message(msg1)
    resp2 = await server.handle_message(msg2)  # Should return duplicate error
    
    assert resp1["result"] is not None
    assert resp2["error"]["code"] == -32000  # Duplicate request error

asyncio.run(test_duplicate())
```

## Integration with Cursor IDE

For Cursor IDE specifically, the robust client can be used in:
- **Custom scripts** that interact with the MCP server
- **Agent integrations** that need reliable MCP communication
- **Testing** to verify server functionality
- **CLI tools** that wrap MCP functionality

The client handles all the edge cases that Cursor's built-in client might encounter, making it perfect for programmatic access.

## Summary

### Server-Side Improvements
- ✅ Detecting and rejecting duplicate requests
- ✅ Cleaning up stale requests automatically
- ✅ Handling client reconnections gracefully
- ✅ Providing debug logging for troubleshooting

### Client-Side Implementation
- ✅ Production-ready `RobustMCPClient` class
- ✅ Automatic message ID tracking and deduplication
- ✅ Timeout handling and stale request cleanup
- ✅ Thread-safe for concurrent use
- ✅ Context manager support for automatic cleanup
- ✅ Comprehensive error handling

### Usage

```python
# Quick start
from mcp.mcp_client import create_client

with create_client(debug=True) as client:
    result = client.call_tool("status", {})
    print(result)
```
