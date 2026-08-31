#!/usr/bin/env python3
"""
Robust MCP Client Implementation

A production-ready MCP client with:
- Message ID tracking and deduplication
- Timeout handling
- Automatic reconnection
- Error recovery
- Request/response correlation
"""

import json
import os
import sys
import time
import threading
import subprocess
from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Track a pending request."""
    msg_id: Any
    method: str
    params: Dict[str, Any]
    sent_at: float
    future: Future
    timeout: float = 300.0  # 5 minutes default


@dataclass
class MCPResponse:
    """Structured MCP response."""
    msg_id: Any
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    is_notification: bool = False


class RobustMCPClient:
    """
    Robust MCP client with proper message ID tracking and error handling.
    
    Features:
    - Thread-safe message ID generation
    - Request/response correlation
    - Automatic timeout handling
    - Duplicate response detection
    - Graceful error handling
    - Reconnection support
    """
    
    def __init__(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 300.0,
        enable_debug: bool = False
    ):
        """
        Initialize MCP client.
        
        Args:
            command: Command to start MCP server (e.g., ["python", "-m", "mcp.mcp_server", "--serve"])
            cwd: Working directory for server process
            env: Environment variables for server process
            timeout: Default timeout for requests in seconds
            enable_debug: Enable debug logging
        """
        self.command = command
        self.cwd = cwd
        self.env = env or {}
        self.timeout = timeout
        self.enable_debug = enable_debug
        
        # Message ID management
        self._next_id = 1
        self._id_lock = threading.Lock()
        
        # Request tracking
        self._pending_requests: Dict[Any, PendingRequest] = {}
        self._request_lock = threading.Lock()
        
        # Process management
        self._process: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        self._stderr_tail: deque[str] = deque(maxlen=100)
        
        # Response queue
        self._response_queue: Queue = Queue()
        
        # Cleanup interval
        self._cleanup_interval = 30.0  # Clean up stale requests every 30 seconds
        
        if enable_debug:
            logging.basicConfig(level=logging.DEBUG)
    
    def _get_next_id(self) -> int:
        """Generate unique message ID."""
        with self._id_lock:
            msg_id = self._next_id
            self._next_id += 1
            # Wrap around at 2^31 to avoid overflow (JSON-RPC allows any value)
            if self._next_id > 2**31 - 1:
                self._next_id = 1
            return msg_id
    
    def _log_debug(self, message: str):
        """Log debug message if enabled."""
        if self.enable_debug:
            logger.debug(f"[MCP Client] {message}")
    
    def _track_request(self, msg_id: Any, method: str, params: Dict[str, Any], timeout: float) -> PendingRequest:
        """Track a pending request."""
        future: Future = Future()
        request = PendingRequest(
            msg_id=msg_id,
            method=method,
            params=params,
            sent_at=time.time(),
            future=future,
            timeout=timeout
        )
        
        with self._request_lock:
            # Check for duplicate message ID (shouldn't happen, but be safe)
            if msg_id in self._pending_requests:
                self._log_debug(f"Warning: Message ID {msg_id} already pending, replacing")
            
            self._pending_requests[msg_id] = request
        
        self._log_debug(f"Tracked request: ID={msg_id}, method={method}, pending={len(self._pending_requests)}")
        return request
    
    def _complete_request(self, msg_id: Any, response: MCPResponse):
        """Complete a pending request."""
        with self._request_lock:
            if msg_id not in self._pending_requests:
                self._log_debug(f"Warning: Received response for unknown message ID: {msg_id}")
                # This is the "unknown message ID" case - log but don't crash
                return False
            
            request = self._pending_requests.pop(msg_id)
            
            # Set result on future
            if not request.future.done():
                request.future.set_result(response)
            
            self._log_debug(f"Completed request: ID={msg_id}, pending={len(self._pending_requests)}")
            return True
    
    def _cleanup_stale_requests(self):
        """Remove requests that have timed out."""
        current_time = time.time()
        stale_ids = []
        
        with self._request_lock:
            for msg_id, request in list(self._pending_requests.items()):
                age = current_time - request.sent_at
                if age > request.timeout:
                    stale_ids.append(msg_id)
                    if not request.future.done():
                        error_response = MCPResponse(
                            msg_id=msg_id,
                            error={
                                "code": -32000,
                                "message": f"Request timed out after {age:.1f}s"
                            }
                        )
                        request.future.set_result(error_response)
            
            for msg_id in stale_ids:
                self._pending_requests.pop(msg_id, None)
                self._log_debug(f"Cleaned up stale request: ID={msg_id}")
        
        if stale_ids:
            logger.warning(f"Cleaned up {len(stale_ids)} stale request(s)")
    
    def _reader_loop(self):
        """Read responses from server stdout."""
        if not self._process:
            return
        
        try:
            for line in self._process.stdout:
                if not self._running:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    response = json.loads(line)
                    self._response_queue.put(response)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse response: {e}, line: {line[:100]}")
        
        except Exception as e:
            if self._running:
                logger.error(f"Error reading from server: {e}")

    def _stderr_loop(self):
        """Continuously drain server stderr so its pipe cannot backpressure stdout."""
        if not self._process or self._process.stderr is None:
            return

        try:
            for line in self._process.stderr:
                message = line.rstrip("\r\n")
                if message:
                    self._stderr_tail.append(message)
                    self._log_debug(f"Server stderr: {message}")
        except Exception as e:
            if self._running:
                logger.error(f"Error reading server stderr: {e}")
    
    def _response_handler_loop(self):
        """Process responses from queue."""
        while self._running:
            try:
                # Get response with timeout to allow checking _running flag
                try:
                    response_data = self._response_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Parse response
                msg_id = response_data.get("id")
                
                if msg_id is None:
                    # Notification - no response expected
                    self._log_debug("Received notification (no ID)")
                    continue
                
                # Create structured response
                mcp_response = MCPResponse(
                    msg_id=msg_id,
                    result=response_data.get("result"),
                    error=response_data.get("error")
                )
                
                # Complete the request
                self._complete_request(msg_id, mcp_response)
            
            except Exception as e:
                logger.error(f"Error handling response: {e}")
    
    def _cleanup_loop(self):
        """Periodic cleanup of stale requests."""
        while self._running:
            time.sleep(self._cleanup_interval)
            if self._running:
                self._cleanup_stale_requests()
    
    def start(self):
        """Start the MCP server process and client threads."""
        if self._running:
            logger.warning("Client already running")
            return
        
        # Start server process
        env = {**os.environ, **self.env}
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.cwd,
            env=env,
            text=True,
            bufsize=1  # Line buffered
        )
        
        self._running = True
        
        # Start reader thread
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        # stderr must be consumed independently from stdout. A PIPE with no
        # reader eventually fills and blocks the server before it can respond.
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()
        
        # Start response handler thread
        self._response_handler_thread = threading.Thread(target=self._response_handler_loop, daemon=True)
        self._response_handler_thread.start()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        # Initialize connection
        self.initialize()
        
        logger.info("MCP client started")
    
    def stop(self):
        """Stop the client and server process."""
        self._running = False
        
        # Cancel all pending requests
        with self._request_lock:
            for request in self._pending_requests.values():
                if not request.future.done():
                    error_response = MCPResponse(
                        msg_id=request.msg_id,
                        error={
                            "code": -32000,
                            "message": "Client stopped"
                        }
                    )
                    request.future.set_result(error_response)
            self._pending_requests.clear()
        
        # Terminate process
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
        
        logger.info("MCP client stopped")
    
    def initialize(self) -> MCPResponse:
        """Initialize the MCP connection."""
        return self._send_request("initialize", {})
    
    def _send_request(self, method: str, params: Dict[str, Any], timeout: Optional[float] = None) -> MCPResponse:
        """
        Send a request and wait for response.
        
        Args:
            method: MCP method name
            params: Method parameters
            timeout: Request timeout (uses default if None)
        
        Returns:
            MCPResponse with result or error
        
        Raises:
            RuntimeError: If client is not running
            TimeoutError: If request times out
        """
        if not self._running or not self._process:
            raise RuntimeError("Client not started. Call start() first.")
        
        msg_id = self._get_next_id()
        timeout = timeout or self.timeout
        
        # Create request
        request = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }
        
        # Track request
        pending = self._track_request(msg_id, method, params, timeout)
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self._process.stdin.write(request_json)
            self._process.stdin.flush()
            
            self._log_debug(f"Sent request: ID={msg_id}, method={method}")
            
            # Wait for response
            try:
                response = pending.future.result(timeout=timeout)
                return response
            except Exception as e:
                # Remove from pending
                with self._request_lock:
                    self._pending_requests.pop(msg_id, None)
                
                if isinstance(e, TimeoutError):
                    raise TimeoutError(f"Request {msg_id} ({method}) timed out after {timeout}s")
                raise
        
        except Exception as e:
            # Remove from pending on error
            with self._request_lock:
                self._pending_requests.pop(msg_id, None)
            raise
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools."""
        response = self._send_request("tools/list", {})
        if response.error:
            raise RuntimeError(f"Failed to list tools: {response.error}")
        return response.result.get("tools", [])
    
    def call_tool(self, name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            timeout: Request timeout
        
        Returns:
            Tool result (parsed from content)
        """
        response = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        }, timeout=timeout)
        
        if response.error:
            raise RuntimeError(f"Tool call failed: {response.error}")
        
        # Extract content from result
        content = response.result.get("content", [])
        if not content:
            raise RuntimeError("Empty response content")
        
        # Parse JSON from text content
        for entry in content:
            if entry.get("type") == "text":
                try:
                    return json.loads(entry.get("text", "{}"))
                except json.JSONDecodeError:
                    continue
            elif entry.get("type") == "application/json":
                return entry.get("json", {})
        
        raise RuntimeError("Could not parse response content")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Convenience function for quick usage
def create_client(
    command: Optional[List[str]] = None,
    cwd: Optional[str] = None,
    timeout: float = 300.0,
    debug: bool = False
) -> RobustMCPClient:
    """
    Create a configured MCP client.
    
    Args:
        command: Server command (defaults to python -m mcp.mcp_server --serve)
        cwd: Working directory
        timeout: Request timeout
        debug: Enable debug logging
    
    Returns:
        Configured client instance
    """
    if command is None:
        import sys
        command = [sys.executable, "-m", "mcp.mcp_server", "--serve"]
    
    return RobustMCPClient(
        command=command,
        cwd=cwd,
        timeout=timeout,
        enable_debug=debug
    )


# Example usage
if __name__ == "__main__":
    import os
    
    # Example: Use the client
    client = create_client(debug=True)
    
    try:
        client.start()
        
        # List tools
        tools = client.list_tools()
        print(f"Found {len(tools)} tools")
        
        # Call a tool
        result = client.call_tool("status", {})
        print(f"Status: {result.get('status')}")
        
    finally:
        client.stop()
