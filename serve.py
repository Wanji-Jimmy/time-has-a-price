"""Serve the checked self-contained demo locally; no third-party dependency."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
from pathlib import Path
import argparse
import threading
import webbrowser
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=0,help="Port number; 0 selects a free port")
    parser.add_argument("--open",action="store_true",help="Open the report in your browser")
    args=parser.parse_args()
    handler=partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).parent/"demo"))
    with ThreadingHTTPServer(("127.0.0.1",args.port),handler) as server:
        url=f"http://127.0.0.1:{server.server_port}/?view=1"
        print(f"Time Has a Price: {url}",flush=True)
        print("Keep this terminal open. Press Ctrl+C to stop.",flush=True)
        if args.open:
            threading.Thread(target=webbrowser.open,args=(url,),daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Demo stopped.",flush=True)
