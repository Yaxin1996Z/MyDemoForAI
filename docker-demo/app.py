"""简单的容器内 Web 服务，返回容器环境信息"""
import http.server
import json
import os
import socket
import datetime


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        info = {
            "message": "Hello from Docker! 🐳",
            "hostname": socket.gethostname(),
            "time": datetime.datetime.now().isoformat(),
            "container": os.environ.get("CONTAINER_NAME", "unknown"),
            "python_version": __import__("sys").version,
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(info, indent=2, ensure_ascii=False).encode())


if __name__ == "__main__":
    port = 8000
    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    print(f"Server running on port {port}...")
    server.serve_forever()
